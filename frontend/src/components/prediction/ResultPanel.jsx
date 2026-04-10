export function ResultPanel({ result, loading, error }) {
  const confidenceScore =
    typeof result?.confidence_score === "number"
      ? Math.max(0, Math.min(1, result.confidence_score))
      : 0

  if (loading) {
    return (
      <section className="rounded-[2rem] border border-cyan-300/15 bg-slate-900/70 p-8 text-center backdrop-blur">
        <div className="mx-auto h-14 w-14 animate-spin rounded-full border-4 border-cyan-300 border-t-transparent" />
        <p className="mt-5 text-lg font-medium text-cyan-100">Analyzing retinal scan...</p>
      </section>
    )
  }

  if (error) {
    return (
      <section className="rounded-[2rem] border border-rose-400/20 bg-rose-400/10 p-6">
        <p className="text-sm font-semibold uppercase tracking-[0.25em] text-rose-200">
          Connection Issue
        </p>
        <p className="mt-3 text-base text-rose-100">{error}</p>
      </section>
    )
  }

  if (!result) {
    return (
      <section className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6 backdrop-blur">
        <p className="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-300/80">
          Awaiting Input
        </p>
        <h2 className="mt-3 text-2xl font-semibold text-white">Results will appear here.</h2>
        <p className="mt-3 max-w-2xl text-base leading-7 text-slate-300">
          After you upload a retinal OCT image and run prediction, this page will display the
          detected disease label, the original scan, and the Grad-CAM attention map.
        </p>
      </section>
    )
  }

  return (
    <section className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6 backdrop-blur">
      <div className="flex flex-col gap-3 border-b border-white/10 pb-5 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-300/80">
            Disease Prediction
          </p>
          <h2 className="mt-2 text-3xl font-semibold text-white sm:text-[2.4rem]">
            {result.prediction}
          </h2>
        </div>
        <div className="flex flex-wrap gap-4">
          <a
            href={result.image_url}
            target="_blank"
            rel="noreferrer"
            className="text-sm font-medium text-cyan-200 transition hover:text-white"
          >
            Open original image
          </a>
          {result.cam_url && (
            <a
              href={result.cam_url}
              target="_blank"
              rel="noreferrer"
              className="text-sm font-medium text-cyan-200 transition hover:text-white"
            >
              Open heatmap
            </a>
          )}
        </div>
      </div>

      <article className="mt-5 rounded-[1.5rem] border border-cyan-300/15 bg-slate-950/70 p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-100/80">
              Confidence Level
            </p>
            <p className="mt-2 text-2xl font-semibold text-white">
              {result.confidence ?? `${Math.round(confidenceScore * 100)}%`}
            </p>
          </div>
          <p className="max-w-md text-sm leading-6 text-slate-400">
            This shows how strongly the model favored the predicted class for the uploaded scan.
          </p>
        </div>
        <div className="mt-4 h-3 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-emerald-300 to-lime-300 transition-all duration-700"
            style={{ width: `${confidenceScore * 100}%` }}
          />
        </div>
      </article>

      {result.ood_assessment && (
        <article
          className={`mt-5 rounded-[1.5rem] border p-5 ${
            result.ood_assessment.is_ood
              ? "border-amber-300/20 bg-amber-300/10"
              : "border-emerald-300/15 bg-emerald-300/10"
          }`}
        >
          <p
            className={`text-sm font-semibold uppercase tracking-[0.28em] ${
              result.ood_assessment.is_ood ? "text-amber-100/90" : "text-emerald-100/90"
            }`}
          >
            Input Reliability
          </p>
          <h3 className="mt-2 text-xl font-semibold text-white">{result.ood_assessment.title}</h3>
          <p className="mt-3 leading-7 text-slate-100">{result.ood_assessment.summary}</p>
          <p className="mt-3 text-sm leading-7 text-slate-300">
            {result.ood_assessment.recommendation}
          </p>
          {result.ood_assessment.reasons?.length > 0 && (
            <div className="mt-3 text-sm text-slate-300">
              {result.ood_assessment.reasons.map((reason) => (
                <p key={reason}>- {reason}</p>
              ))}
            </div>
          )}
        </article>
      )}

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <article className="rounded-[1.5rem] border border-white/10 bg-slate-950/70 p-4">
          <h3 className="text-lg font-semibold text-white">Original Scan</h3>
          <img
            src={result.image_url}
            alt="Original OCT scan"
            className="mt-3 h-[18rem] w-full rounded-[1.25rem] object-contain lg:h-[20rem]"
          />
        </article>

        {result.cam_url && (
          <article className="rounded-[1.5rem] border border-white/10 bg-slate-950/70 p-4">
            <h3 className="text-lg font-semibold text-white">Heatmap Output</h3>
            <img
              src={result.cam_url}
              alt="Grad-CAM heatmap"
              className="mt-3 h-[18rem] w-full rounded-[1.25rem] object-contain lg:h-[20rem]"
            />
          </article>
        )}
      </div>

      {result.heatmap_interpretation && (
        <article className="mt-5 rounded-[1.5rem] border border-cyan-300/15 bg-cyan-300/5 p-5">
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-100/80">
            Heatmap Interpretation
          </p>
          <h3 className="mt-2 text-xl font-semibold text-white sm:text-2xl">
            {result.heatmap_interpretation.title}
          </h3>
          <p className="mt-3 leading-7 text-slate-200">
            {result.heatmap_interpretation.summary}
          </p>
          <p className="mt-3 text-sm leading-7 text-slate-400">
            {result.heatmap_interpretation.clinical_note}
          </p>
        </article>
      )}
    </section>
  )
}
