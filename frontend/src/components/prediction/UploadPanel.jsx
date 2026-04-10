export function UploadPanel({ file, loading, onFileChange, onSubmit }) {
  return (
    <section className="rounded-[2rem] border border-white/10 bg-white/[0.05] p-5 backdrop-blur">
      <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-200/80">
        Prediction Workspace
      </p>
      <h1 className="mt-3 font-serif text-3xl font-semibold text-white sm:text-[2.4rem]">
        Upload a retinal OCT image for analysis.
      </h1>
      <p className="mt-3 max-w-2xl text-base leading-7 text-slate-300">
        This page keeps the original prediction logic intact and presents the result in a cleaner,
        more focused clinical screening layout.
      </p>

      <form onSubmit={onSubmit} className="mt-5 grid gap-4 lg:grid-cols-[1fr_auto]">
        <label className="group flex cursor-pointer items-center justify-between gap-4 rounded-[1.5rem] border border-dashed border-white/20 bg-slate-950/70 px-5 py-3 transition hover:border-cyan-300/40 hover:bg-slate-950">
          <div>
            <p className="text-sm font-semibold text-white">
              {file ? file.name : "Choose OCT scan"}
            </p>
            <p className="mt-1 text-sm text-slate-400">
              PNG, JPG, or any image format accepted by the existing backend.
            </p>
          </div>
          <span className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-200 transition group-hover:bg-white/10">
            Browse
          </span>
          <input
            type="file"
            accept="image/*"
            onChange={onFileChange}
            required
            className="hidden"
          />
        </label>

        <button
          type="submit"
          disabled={!file || loading}
          className={`rounded-full px-6 py-3 text-sm font-semibold transition ${
            !file || loading
              ? "cursor-not-allowed bg-slate-700 text-slate-400"
              : "bg-white text-slate-950 hover:bg-slate-200"
          }`}
        >
          {loading ? "Analyzing..." : "Analyze Scan"}
        </button>
      </form>
    </section>
  )
}
