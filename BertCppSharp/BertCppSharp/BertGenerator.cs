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
        // var embeddings = new float[EmbeddingSize!.Value];

        // use the single text api
        // BertCppNative.bert_encode(_bertCtx, _threadCount, text, embeddings);

        // // use the separate tokenization & eval api
        // var tokens = new int[MaxSequenceLength!.Value];
        // var nTokens = 0;
        // unsafe {
        //     // fixed (int* pTokens = tokens) {
        //     //     BertCppNative.bert_tokenize(_bertCtx, text, pTokens, &nTokens, MaxSequenceLength.Value);
        //     // }
        //     BertCppNative.bert_tokenize(_bertCtx, text, tokens, &nTokens, MaxSequenceLength.Value);
        // }
        //
        // BertCppNative.bert_eval(_bertCtx, _threadCount, tokens, nTokens, embeddings);

        var tokens = Tokenize(text);
        var embeddings = Eval(tokens);

        return embeddings;
    }
    
    public async Task<float[]> EmbedAsync(string text) {
        return await Task.Run(() => Embed(text));
    }

    public int[] Tokenize(string text) {
        var tokens = new int[MaxSequenceLength!.Value];
        var nTokens = 0;
        unsafe {
            // fixed (int* pTokens = tokens) {
            //     BertCppNative.bert_tokenize(_bertCtx, text, pTokens, &nTokens, MaxSequenceLength.Value);
            // }
            BertCppNative.bert_tokenize(_bertCtx, text, tokens, &nTokens, MaxSequenceLength.Value);
        }
        return tokens[..nTokens];
    }
    
    public async Task<int[]> TokenizeAsync(string text) {
        return await Task.Run(() => Tokenize(text));
    }

    public float[] Eval(int[] tokens) {
        var embeddings = new float[EmbeddingSize!.Value];
        BertCppNative.bert_eval(_bertCtx, _threadCount, tokens, tokens.Length, embeddings);
        return embeddings;
    }
    
    public async Task<float[]> EvalAsync(int[] tokens) {
        return await Task.Run(() => Eval(tokens));
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