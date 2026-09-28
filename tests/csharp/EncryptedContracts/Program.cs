using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using DeepVisionStudio;

// Reuse the native fixtures; no test framework, downloads, or model weights.
if (args.Length != 1)
    throw new ArgumentException("Usage: EncryptedContracts <native-build/test-models>");
var root = Path.GetFullPath(args[0]);
var names = JsonSerializer.Deserialize<string[]>(File.ReadAllText(Path.Combine(root, "encrypted-tests.json")))!;
var key = File.ReadAllBytes(Path.Combine(root, "example-only.key"));
try
{
    Require(names.Length != 0, "No encrypted fixtures were generated");
    var covered = new HashSet<string>();
    foreach (var name in names)
    {
        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, name + ".json")));
        var backend = config.RootElement.GetProperty("backend").GetString();
        var task = backend == "sam2" ? "sam2" : config.RootElement.GetProperty("task").GetString()!;
        covered.Add(task);
        // Repeated construction/disposal also tests continued operation after
        // every previous native session and SAM image context was released.
        for (var cycle = 0; cycle < 3; ++cycle)
        {
            using var plain = VisionSession.Open(Path.Combine(root, name + ".json"));
            using var encrypted = OpenAndEraseKey(Path.Combine(root, name + ".dvsenc"), key);
            for (var sample = 0; sample < 5; ++sample)
            {
                var pixels = Enumerable.Repeat((byte)(sample * 63), 37 * 24).ToArray();
                CompareInference(plain, encrypted, task, pixels, sample);
            }
            encrypted.Dispose();
            encrypted.Dispose(); // SafeHandle disposal must be idempotent.
            Require(encrypted.IsClosed && encrypted.IsInvalid, "Disposed session remains valid");
            Expect<ObjectDisposedException>(() => encrypted.InferClassification(new byte[1], 1, 1, 1),
                                             "Inference on disposed session");
        }
        Console.WriteLine($"PASS managed byte-exact: {name} (3 load/dispose cycles)");
    }
    Require(covered.SetEquals(new[] { "classify", "segment", "detect", "anomaly", "sam2" }),
            "Fixtures must cover classification, segmentation, detection, anomaly, and SAM2");

    var validPath = Path.Combine(root, "classify.dvsenc");
    var wrongKey = (byte[])key.Clone();
    wrongKey[0] ^= 1;
    try
    {
        Expect<InvalidOperationException>(() => { using var unexpected = VisionSession.OpenEncrypted(validPath, wrongKey); },
                                          "Wrong key accepted");
    }
    finally { CryptographicOperations.ZeroMemory(wrongKey); }
    Expect<ArgumentException>(() => { using var unexpected = VisionSession.OpenEncrypted(validPath, new byte[31]); },
                              "Short key accepted");
    Expect<ArgumentOutOfRangeException>(() => { using var unexpected = VisionSession.OpenEncrypted(validPath, key, -2); },
                                        "Invalid thread override accepted");
    var invalidNames = JsonSerializer.Deserialize<string[]>(
        File.ReadAllText(Path.Combine(root, "encrypted-invalid-tests.json")))!;
    Require(invalidNames.Contains("tampered") && invalidNames.Contains("truncated"),
            "Missing modified-file fixtures");
    foreach (var name in invalidNames)
        Expect<InvalidOperationException>(() => {
            using var unexpected = VisionSession.OpenEncrypted(Path.Combine(root, name + ".dvsenc"), key);
        }, $"Malformed package accepted: {name}");

    // Native create failures must not corrupt the following successful load.
    using (var recovered = OpenAndEraseKey(validPath, key))
        Require(recovered.InferClassification(new byte[37 * 24], 32, 24, 1, 37).Probabilities.Length == 2,
                "A failed load prevented later inference");
    Console.WriteLine($"PASS managed authentication/errors: wrong key, short key, thread override, {invalidNames.Length} malformed files");
}
finally { CryptographicOperations.ZeroMemory(key); }

static VisionSession OpenAndEraseKey(string path, byte[] fixtureKey)
{
    var callerKey = (byte[])fixtureKey.Clone();
    VisionSession? session = null;
    try
    {
        session = VisionSession.OpenEncrypted(path, callerKey);
        Require(callerKey.AsSpan().SequenceEqual(fixtureKey), "OpenEncrypted changed the caller-owned key");
        return session;
    }
    catch { session?.Dispose(); throw; }
    finally { CryptographicOperations.ZeroMemory(callerKey); }
}

static void CompareInference(VisionSession plain, VisionSession encrypted, string task, byte[] pixels, int sample)
{
    switch (task)
    {
        case "classify":
            var ac = plain.InferClassification(pixels, 32, 24, 1, 37);
            var bc = encrypted.InferClassification(pixels, 32, 24, 1, 37);
            Require(ac.ClassId == bc.ClassId && ac.ClassName == bc.ClassName, "Classification metadata differs");
            EqualFloat(ac.Confidence, bc.Confidence);
            EqualFloats(ac.Probabilities, bc.Probabilities);
            break;
        case "segment":
            EqualMask(plain.InferSegmentation(pixels, 32, 24, 1, 37),
                      encrypted.InferSegmentation(pixels, 32, 24, 1, 37));
            break;
        case "detect":
            var ad = plain.InferDetection(pixels, 32, 24, 1, 37).Detections;
            var bd = encrypted.InferDetection(pixels, 32, 24, 1, 37).Detections;
            Require(ad.Length == bd.Length, "Detection count differs");
            for (var i = 0; i < ad.Length; ++i)
            {
                Require(ad[i].ClassId == bd[i].ClassId, "Detection class differs");
                EqualFloats(new[] { ad[i].X1, ad[i].Y1, ad[i].X2, ad[i].Y2, ad[i].Confidence },
                            new[] { bd[i].X1, bd[i].Y1, bd[i].X2, bd[i].Y2, bd[i].Confidence });
            }
            break;
        case "anomaly":
            var aa = plain.InferAnomaly(pixels, 32, 24, 1, 37);
            var ba = encrypted.InferAnomaly(pixels, 32, 24, 1, 37);
            Require(aa.Width == ba.Width && aa.Height == ba.Height && aa.IsAnomalous == ba.IsAnomalous,
                    "Anomaly metadata differs");
            EqualFloat(aa.Score, ba.Score);
            EqualFloat(aa.Threshold, ba.Threshold);
            EqualFloats(aa.Map, ba.Map);
            break;
        case "sam2":
            using (var ap = plain.EncodeSam(pixels, 32, 24, 1, 37))
            using (var bp = encrypted.EncodeSam(pixels, 32, 24, 1, 37))
            {
                var prompt = new SamPrompt(new float[] { 16, 12 }, new[] { 1 },
                    sample == 1 ? new float[] { 4, 4, 24, 20 } : null,
                    sample == 2 ? new float[] { 1, 0, 0, 1 } : null, 2, 2);
                EqualMask(plain.SegmentSam(ap, prompt), encrypted.SegmentSam(bp, prompt));
                if (sample == 0)
                {
                    EqualMask(plain.AutomaticSam(ap, 2, 2), encrypted.AutomaticSam(bp, 2, 2));
                    Expect<ArgumentException>(() => encrypted.SegmentSam(ap, prompt), "Foreign SAM context accepted");
                }
                bp.Dispose();
                Require(bp.IsClosed && bp.IsInvalid, "Disposed SAM context remains valid");
                Expect<ArgumentException>(() => encrypted.SegmentSam(bp, prompt), "Disposed SAM context accepted");
            }
            break;
        default: throw new InvalidOperationException($"Unknown fixture task: {task}");
    }
}

static void EqualMask(SegmentationResult a, SegmentationResult b)
{
    Require(a.Width == b.Width && a.Height == b.Height && a.Classes == b.Classes, "Mask metadata differs");
    Require(a.Mask.AsSpan().SequenceEqual(b.Mask), "Mask bytes differ");
}

static void EqualFloat(float a, float b) =>
    Require(BitConverter.SingleToInt32Bits(a) == BitConverter.SingleToInt32Bits(b), "Float bits differ");

static void EqualFloats(float[] a, float[] b) =>
    Require(MemoryMarshal.AsBytes(a.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(b.AsSpan())), "Float array bytes differ");

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

static void Expect<T>(Action action, string message) where T : Exception
{
    try { action(); }
    catch (T) { return; }
    throw new InvalidOperationException(message);
}
