namespace BertCppSharp;

public class BertGenerator : IDisposable {
    private IntPtr _bertCtx;
    private readonly int _threadCount;
    public string? ModelFile { get; private set; }
    public int? MaxSequenceLength { get; private set; }
    public int? EmbeddingSize { get; private set; }

    public BertGenerator(int threads = 0) {
        if (threads == 0) {
            threads = Environment.ProcessorCount;
        }
        _threadCount = threads;
    }

    public async Task LoadModelAsync(string modelFile) {
        await Task.Run(() => LoadModel(modelFile));
    }
    public void LoadModel(string modelFile) {
        _bertCtx = BertCppNative.bert_load_from_file(modelFile);
        ModelFile = modelFile;

        MaxSequenceLength = BertCppNative.bert_n_max_tokens(_bertCtx);
        EmbeddingSize = BertCppNative.bert_n_embd(_bertCtx);
    }

    public float[] Embed(string text) {
        var embeddings = new float[EmbeddingSize!.Value];
        BertCppNative.bert_encode(_bertCtx, _threadCount, text, embeddings);
        return embeddings;
    }

    private void FreeContext() {
        BertCppNative.bert_free(_bertCtx);
        _bertCtx = IntPtr.Zero;
    }

    public void Dispose() {
        if (_bertCtx != IntPtr.Zero) {
            FreeContext();
        }
    }
}