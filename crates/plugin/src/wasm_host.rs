//! WebAssembly Sandboxed Plugin Host using Wasmtime.
//!
//! Enforces isolation, strict memory quotas, and zero-panic safety boundaries.
//! Untrusted WASM plugins cannot access host files, networks, or memory outside
//! their allocated linear memory space.
//!
//! Same API as the PR, with the sandbox actually enforcing what it documents:
//! a fuel budget per frame (a plugin that loops forever stops instead of
//! hanging FilmCraft), a memory ceiling taken from the manifest's
//! `memory_limit_mb`, and one long-lived instance per plugin so a guest can
//! keep state between frames (a previous-frame cache, for instance) instead of
//! being re-instantiated for every image.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::error::PluginError;
use crate::manifest::PluginManifest;
use crate::traits::{FilmCraftPlugin, VideoEffectPlugin, validate_frame_bounds};
use crate::wasm_abi::{ALLOC_SYMBOL, DEALLOC_SYMBOL, GET_MANIFEST_SYMBOL, PROCESS_VIDEO_FRAME_SYMBOL, unpack_ptr_len};

#[cfg(feature = "wasm")]
use wasmtime::{Config, Engine, Linker, Memory, Module, Store, TypedFunc};

/// Instructions the sandbox may run for one frame, whatever its size: enough
/// for a heavy pixel effect at 8K, small enough that a runaway loop stops
/// within seconds.
#[cfg(feature = "wasm")]
const FUEL_BASE: u64 = 50_000_000;
/// Instructions a sandbox may spend per pixel on top of [`FUEL_BASE`].
#[cfg(feature = "wasm")]
const FUEL_PAR_PIXEL: u64 = 600;

/// Guest memory ceiling, applied both to the module's declared maximum and to
/// every `memory.grow` a plugin requests.
#[cfg(feature = "wasm")]
struct SandboxLimits {
    max_bytes: usize,
}

#[cfg(feature = "wasm")]
impl wasmtime::ResourceLimiter for SandboxLimits {
    fn memory_growing(&mut self, _current: usize, desired: usize, maximum: Option<usize>) -> wasmtime::Result<bool> {
        let ceiling = maximum.map_or(self.max_bytes, |m| m.min(self.max_bytes));
        Ok(desired <= ceiling)
    }

    fn table_growing(&mut self, _current: usize, desired: usize, maximum: Option<usize>) -> wasmtime::Result<bool> {
        let ceiling = maximum.unwrap_or(10_000);
        Ok(desired <= ceiling)
    }
}

/// Shared Wasmtime engine instance with optimal JIT compilation settings.
#[derive(Clone)]
pub struct WasmHostEngine {
    #[cfg(feature = "wasm")]
    engine: Engine,
}

impl WasmHostEngine {
    /// Creates a new WebAssembly host engine configured for FilmCraft.
    pub fn new() -> Result<Self, PluginError> {
        #[cfg(feature = "wasm")]
        {
            let mut config = Config::new();
            config.cranelift_opt_level(wasmtime::OptLevel::Speed);
            config.wasm_simd(true);
            config.wasm_bulk_memory(true);
            config.wasm_multi_memory(false);
            // The fuel budget is what makes a runaway plugin stop.
            config.consume_fuel(true);

            let engine = Engine::new(&config).map_err(|e| PluginError::InitializationFailed { id: "wasm_engine".into(), reason: e.to_string() })?;

            Ok(Self { engine })
        }
        #[cfg(not(feature = "wasm"))]
        {
            Ok(Self {})
        }
    }
}

/// One instantiated sandbox, kept alive between frames so a plugin can carry
/// state (its own previous-frame cache, for one).
#[cfg(feature = "wasm")]
struct Sandbox {
    store: Store<SandboxLimits>,
    memory: Memory,
    alloc: TypedFunc<u32, u32>,
    dealloc: TypedFunc<(u32, u32), ()>,
    process: TypedFunc<(u32, u32, u32, u32, u32, u32), i32>,
}

/// A sandboxed video effect plugin backed by a compiled `.wasm` WebAssembly binary.
pub struct WasmVideoEffectProxy {
    manifest: PluginManifest,
    #[cfg(feature = "wasm")]
    engine: Engine,
    #[cfg(feature = "wasm")]
    module: Module,
    #[cfg(feature = "wasm")]
    sandbox: Mutex<Option<Sandbox>>,
}

impl WasmVideoEffectProxy {
    /// Instantiates and inspects a WASM plugin binary.
    ///
    /// Reads and parses the declarative manifest directly from the WASM boundary.
    pub fn from_wasm_bytes(host: &WasmHostEngine, wasm_bytes: &[u8]) -> Result<Self, PluginError> {
        #[cfg(feature = "wasm")]
        {
            let engine = host.engine.clone();
            let module = Module::new(&engine, wasm_bytes)
                .map_err(|e| PluginError::InitializationFailed { id: "wasm_module".into(), reason: format!("failed to compile WASM binary: {e}") })?;

            // Read the manifest from a throwaway instance.
            let mut store = Store::new(&engine, SandboxLimits { max_bytes: 128 * 1024 * 1024 });
            let linker = Linker::new(&engine);
            let instance = linker
                .instantiate(&mut store, &module)
                .map_err(|e| PluginError::InitializationFailed { id: "wasm_instance".into(), reason: format!("failed to instantiate WASM plugin: {e}") })?;

            let memory = instance
                .get_memory(&mut store, "memory")
                .ok_or_else(|| PluginError::InitializationFailed { id: "wasm_memory".into(), reason: "WASM module does not export linear 'memory'".into() })?;

            let get_manifest_fn: TypedFunc<(), u64> = instance.get_typed_func(&mut store, GET_MANIFEST_SYMBOL).map_err(|e| {
                PluginError::InitializationFailed { id: "wasm_manifest_fn".into(), reason: format!("missing export '{GET_MANIFEST_SYMBOL}': {e}") }
            })?;

            let _ = store.set_fuel(FUEL_BASE);
            let packed = get_manifest_fn
                .call(&mut store, ())
                .map_err(|e| PluginError::InitializationFailed { id: "wasm_manifest_call".into(), reason: format!("failed to retrieve manifest: {e}") })?;

            let (ptr, len) = unpack_ptr_len(packed);
            if len == 0 || len > 1024 * 1024 {
                return Err(PluginError::Manifest(format!("manifest length {len} is not plausible")));
            }
            let mut manifest_bytes = vec![0u8; len as usize];
            memory.read(&store, ptr as usize, &mut manifest_bytes).map_err(|e| PluginError::InitializationFailed {
                id: "wasm_memory_read".into(),
                reason: format!("failed to read manifest from memory: {e}"),
            })?;

            let manifest_str = std::str::from_utf8(&manifest_bytes).map_err(|e| PluginError::Manifest(format!("invalid UTF-8 manifest: {e}")))?;

            let manifest = PluginManifest::from_json(manifest_str)?;

            Ok(Self { manifest, engine, module, sandbox: Mutex::new(None) })
        }
        #[cfg(not(feature = "wasm"))]
        {
            let _ = (host, wasm_bytes);
            Err(PluginError::Unsupported("WASM support is disabled in this build (feature 'wasm' required)".into()))
        }
    }

    /// Upper bound on guest memory, from the manifest, in bytes.
    #[cfg(feature = "wasm")]
    fn memory_ceiling(&self) -> usize {
        let mo = self.manifest.capabilities.memory_limit_mb.clamp(16, 2048) as usize;
        mo * 1024 * 1024
    }

    /// Runs `action` on the plugin's sandbox, creating it on first use.
    #[cfg(feature = "wasm")]
    fn with_instance<R>(&self, action: impl FnOnce(&mut Sandbox) -> Result<R, PluginError>) -> Result<R, PluginError> {
        let mut guard = self.sandbox.lock().map_err(|_| PluginError::ExecutionFailed("the plugin sandbox lock is poisoned".into()))?;

        if guard.is_none() {
            *guard = Some(self.instantiate()?);
        }

        let result = match guard.as_mut() {
            Some(sandbox) => action(sandbox),
            None => Err(PluginError::ExecutionFailed("plugin sandbox unavailable".into())),
        };

        if result.is_err() {
            // After a trap or a failure the instance is worthless: the next
            // frame starts from a fresh sandbox.
            *guard = None;
        }
        result
    }

    #[cfg(feature = "wasm")]
    fn instantiate(&self) -> Result<Sandbox, PluginError> {
        let mut store = Store::new(&self.engine, SandboxLimits { max_bytes: self.memory_ceiling() });
        let ceiling = self.memory_ceiling();
        store.limiter(move |limiteur| limiteur);
        let _ = store.set_fuel(FUEL_BASE + 1_000_000);

        let linker = Linker::new(&self.engine);
        let instance =
            linker.instantiate(&mut store, &self.module).map_err(|e| PluginError::ExecutionFailed(format!("failed to instantiate plugin sandbox: {e}")))?;

        let memory = instance.get_memory(&mut store, "memory").ok_or_else(|| PluginError::ExecutionFailed("WASM sandbox missing linear 'memory'".into()))?;

        let alloc = instance.get_typed_func(&mut store, ALLOC_SYMBOL).map_err(|e| PluginError::ExecutionFailed(format!("missing '{ALLOC_SYMBOL}': {e}")))?;
        let dealloc =
            instance.get_typed_func(&mut store, DEALLOC_SYMBOL).map_err(|e| PluginError::ExecutionFailed(format!("missing '{DEALLOC_SYMBOL}': {e}")))?;
        let process = instance
            .get_typed_func(&mut store, PROCESS_VIDEO_FRAME_SYMBOL)
            .map_err(|e| PluginError::ExecutionFailed(format!("missing '{PROCESS_VIDEO_FRAME_SYMBOL}': {e}")))?;

        // A plugin that grew beyond the ceiling while instantiating its own
        // data is refused now rather than later.
        let bytes = memory.data_size(&store);
        if bytes > ceiling {
            return Err(PluginError::ExecutionFailed(format!("plugin memory {bytes} exceeds the {} MiB it declared", ceiling / (1024 * 1024))));
        }

        Ok(Sandbox { store, memory, alloc, dealloc, process })
    }
}

impl FilmCraftPlugin for WasmVideoEffectProxy {
    fn id(&self) -> &str {
        &self.manifest.id
    }

    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }
}

impl VideoEffectPlugin for WasmVideoEffectProxy {
    fn process_frame(
        &self,
        in_buffer: &[u8],
        out_buffer: &mut [u8],
        width: u32,
        height: u32,
        parameters: &BTreeMap<String, serde_json::Value>,
    ) -> Result<(), PluginError> {
        let expected_bytes = validate_frame_bounds(in_buffer, out_buffer, width, height)?;

        #[cfg(feature = "wasm")]
        {
            let params_json =
                serde_json::to_string(parameters).map_err(|e| PluginError::InvalidParameter { name: "parameters".into(), reason: e.to_string() })?;
            let params_bytes = params_json.as_bytes();
            let pixels = u64::from(width) * u64::from(height);
            let fuel = FUEL_BASE + pixels * FUEL_PAR_PIXEL;
            let bytes = expected_bytes as u32;

            self.with_instance(|sandbox| {
                let _ = sandbox.store.set_fuel(fuel);
                let store = &mut sandbox.store;

                let in_ptr = sandbox.alloc.call(&mut *store, bytes).map_err(|e| PluginError::ExecutionFailed(format!("alloc in_buffer failed: {e}")))?;
                let out_ptr = sandbox.alloc.call(&mut *store, bytes).map_err(|e| PluginError::ExecutionFailed(format!("alloc out_buffer failed: {e}")))?;
                let params_ptr = sandbox
                    .alloc
                    .call(&mut *store, params_bytes.len().max(1) as u32)
                    .map_err(|e| PluginError::ExecutionFailed(format!("alloc params failed: {e}")))?;

                if in_ptr == 0 || out_ptr == 0 || params_ptr == 0 {
                    return Err(PluginError::ExecutionFailed("the plugin refused to allocate memory".into()));
                }

                sandbox
                    .memory
                    .write(&mut *store, in_ptr as usize, &in_buffer[..expected_bytes])
                    .map_err(|e| PluginError::ExecutionFailed(format!("memory write input failed: {e}")))?;
                sandbox
                    .memory
                    .write(&mut *store, params_ptr as usize, params_bytes)
                    .map_err(|e| PluginError::ExecutionFailed(format!("memory write params failed: {e}")))?;

                let status_code = sandbox
                    .process
                    .call(&mut *store, (in_ptr, out_ptr, width, height, params_ptr, params_bytes.len() as u32))
                    .map_err(|e| PluginError::ExecutionFailed(format!("WASM sandbox execution trap: {e}")))?;

                if status_code != 0 {
                    let _ = sandbox.dealloc.call(&mut *store, (in_ptr, bytes));
                    let _ = sandbox.dealloc.call(&mut *store, (out_ptr, bytes));
                    let _ = sandbox.dealloc.call(&mut *store, (params_ptr, params_bytes.len() as u32));
                    return Err(PluginError::ExecutionFailed(format!("plugin returned status error code: {status_code}")));
                }

                sandbox
                    .memory
                    .read(&store, out_ptr as usize, &mut out_buffer[..expected_bytes])
                    .map_err(|e| PluginError::ExecutionFailed(format!("memory read output failed: {e}")))?;

                let _ = sandbox.dealloc.call(&mut *store, (in_ptr, bytes));
                let _ = sandbox.dealloc.call(&mut *store, (out_ptr, bytes));
                let _ = sandbox.dealloc.call(&mut *store, (params_ptr, params_bytes.len() as u32));

                Ok(())
            })
        }
        #[cfg(not(feature = "wasm"))]
        {
            let _ = (in_buffer, out_buffer, width, height, parameters, expected_bytes);
            Err(PluginError::Unsupported("WASM support is disabled in this build".into()))
        }
    }
}

#[cfg(all(test, feature = "wasm"))]
mod sandbox_tests {
    use super::*;

    /// A 1 MiB ceiling for the test guests below.
    const TEST_CEILING: usize = 1024 * 1024;

    fn store_with_limits(engine: &Engine) -> Store<SandboxLimits> {
        let mut store = Store::new(engine, SandboxLimits { max_bytes: TEST_CEILING });
        store.limiter(|limits| limits);
        store.set_fuel(FUEL_BASE).expect("the engine must allow fuel metering");
        store
    }

    /// A plugin that loops forever must be stopped by the fuel budget instead of
    /// hanging the host, which is what `Config::consume_fuel` buys us.
    #[test]
    fn runaway_guest_is_stopped_by_the_fuel_budget() {
        let engine = WasmHostEngine::new().expect("host engine").engine;
        let module = Module::new(&engine, r#"(module (func (export "spin") (loop $l br $l)))"#).expect("test module compiles");
        let mut store = store_with_limits(&engine);
        let instance = Linker::new(&engine).instantiate(&mut store, &module).expect("instantiate");
        let spin = instance.get_typed_func::<(), ()>(&mut store, "spin").expect("export 'spin'");

        let error = spin.call(&mut store, ()).expect_err("an endless loop must trap");
        let trap = error.root_cause().downcast_ref::<wasmtime::Trap>().copied();
        assert_eq!(trap, Some(wasmtime::Trap::OutOfFuel), "a runaway guest must stop on fuel exhaustion, got: {error:?}");
    }

    /// `memory_limit_mb` is enforced on every `memory.grow`, not merely parsed.
    #[test]
    fn guest_memory_growth_stops_at_the_manifest_ceiling() {
        let engine = WasmHostEngine::new().expect("host engine").engine;
        let module = Module::new(
            &engine,
            r#"(module
                 (memory (export "memory") 1)
                 (func (export "grow") (param i32) (result i32) local.get 0 memory.grow))"#,
        )
        .expect("test module compiles");
        let mut store = store_with_limits(&engine);
        let instance = Linker::new(&engine).instantiate(&mut store, &module).expect("instantiate");
        let memory = instance.get_memory(&mut store, "memory").expect("export 'memory'");
        let grow = instance.get_typed_func::<u32, i32>(&mut store, "grow").expect("export 'grow'");

        // One 64 KiB page of headroom is granted (the call returns the previous
        // size, so 1), a hundred pages are refused with -1.
        assert_eq!(grow.call(&mut store, 1).expect("first growth"), 1);
        assert_eq!(grow.call(&mut store, 100).expect("second growth"), -1);
        assert!(memory.data_size(&store) <= TEST_CEILING, "guest memory {} bytes", memory.data_size(&store));
    }
}
