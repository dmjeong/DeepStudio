# Managed encrypted-model contracts

Build the native SDK and run its `generate_deployment_fixtures` CTest first.
The console uses those same fixtures to compare every managed result field
except timings, with three session load/dispose cycles and five padded images
per model. SAM2 covers points, boxes, masks, automatic masks, and context disposal.
It also checks wrong keys, modified/malformed files, key ownership, and recovery
after initialization errors.

On Windows, set `DEEP_VISION_NATIVE_RUNTIME_DIR` to the native DLL directory
(including its ORT/OpenCV dependencies), then run:

```bat
dotnet run --project tests/csharp/EncryptedContracts/EncryptedContracts.csproj -c Release -- build/test-models
```

On macOS/Linux, expose the native library directory through `DYLD_LIBRARY_PATH`
or `LD_LIBRARY_PATH` respectively. Exit code zero means all contracts passed.
