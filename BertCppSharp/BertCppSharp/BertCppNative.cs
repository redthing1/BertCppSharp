using System.Runtime.InteropServices;

namespace BertCppSharp;

public static class BertCppNative {
    public const string NativeLibrary = "bert";

    //     struct bert_ctx;
    //
    //     alias bert_vocab_id = int;
    //
    //     bert_ctx* bert_load_from_file(const (char)* fname);
    //     void bert_free(bert_ctx* ctx);
    //

    [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr bert_load_from_file([MarshalAs(UnmanagedType.LPStr)] string fname);

    [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void bert_free(IntPtr ctx);

    // // Main api, does both tokenizing and evaluation
    //
    //     void bert_encode(
    //         bert_ctx* ctx,
    //         int n_threads, 
    //     const (char)* texts, 
    //     float* embeddings);
    //

    [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void bert_encode(IntPtr ctx, int n_threads, [MarshalAs(UnmanagedType.LPStr)] string text, float[] embeddings);

    // // // n_batch_size - how many to process at a time
    // // // n_inputs     - total size of texts and embeddings arrays
    // //     void bert_encode_batch(
    // //         bert_ctx* ctx,
    // //         int n_threads,
    // //         int n_batch_size,
    // //         int n_inputs, 
    // //     const (char*)* texts, 
    // //     float** embeddings);
    // //
    //
    // [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    // public static extern void bert_encode_batch(IntPtr ctx, int n_threads, int n_batch_size, int n_inputs, [MarshalAs(UnmanagedType.LPStr)] string[] texts, float[] embeddings);

    // // Api for separate tokenization & eval
    //
    //     void bert_tokenize(
    //         bert_ctx* ctx, 
    //
    //     const (char)* text,
    //         bert_vocab_id* tokens,
    //     int* n_tokens,
    //     int n_max_tokens);
    //
    
    [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe void bert_tokenize(IntPtr ctx, [MarshalAs(UnmanagedType.LPStr)] string text, int[] tokens, int* n_tokens, int n_max_tokens);

    //     void bert_eval(
    //         bert_ctx* ctx,
    //         int n_threads,
    //         bert_vocab_id* tokens,
    //         int n_tokens,
    //         float* embeddings
    //     );
    //
    
    [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern void bert_eval(IntPtr ctx, int n_threads, int[] tokens, int n_tokens, float[] embeddings);

    // // // NOTE: for batch processing the longest input must be first
    // //     void bert_eval_batch(
    // //         bert_ctx* ctx,
    // //         int n_threads,
    // //         int n_batch_size,
    // //         bert_vocab_id** batch_tokens,
    // //         int* n_tokens,
    // //         float** batch_embeddings
    // //     );
    // //
    //
    // [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    // public static extern unsafe void bert_eval_batch(IntPtr ctx, int n_threads, int n_batch_size, int[]* batch_tokens, int* n_tokens, float[]* batch_embeddings);

    //     int bert_n_embd(bert_ctx* ctx);

    [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern int bert_n_embd(IntPtr ctx);

    //     int bert_n_max_tokens(bert_ctx* ctx);

    [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern int bert_n_max_tokens(IntPtr ctx);

    //
    //     const (char)* bert_vocab_id_to_token (bert_ctx* ctx, bert_vocab_id id);   

    [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr bert_vocab_id_to_token(IntPtr ctx, int id);
}