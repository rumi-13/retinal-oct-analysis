export function ResultPanel({ result, loading, error }) {
  const confidenceScore =
    typeof result?.original_confidence === "number"
      ? Math.max(0, Math.min(1, result.original_confidence))
      : typeof result?.confidence_score === "number"
        ? Math.max(0, Math.min(1, result.confidence_score))
      : 0

  const analysisImages = [
    {
      key: "uploaded_image",
      title: "Uploaded Scan",
      url: result?.uploaded_image ?? result?.image_url,
      alt: "Uploaded OCT scan",
    },
    {
      key: "standard_gradcam_url",
      title: "Standard Grad-CAM",
      url: result?.standard_gradcam_url,
      alt: "Standard Grad-CAM output",
    },
    {
      key: "layer2_heatmap_url",
      title: "Layer 2 Heatmap",
      url: result?.layer2_heatmap_url,
      alt: "Layer 2 heatmap",
    },
    {
      key: "layer3_heatmap_url",
      title: "Layer 3 Heatmap",
      url: result?.layer3_heatmap_url,
      alt: "Layer 3 heatmap",
    },
    {
      key: "layer4_heatmap_url",
      title: "Layer 4 Heatmap",
      url: result?.layer4_heatmap_url,
      alt: "Layer 4 heatmap",
    },
    {
      key: "fused_cam_url",
      title: "Adaptive Fused CAM Overlay",
      url: result?.fused_cam_url,
      alt: "Fused CAM overlay",
    },
    {
      key: "fused_heatmap_url",
      title: "Adaptive Fused Heatmap",
      url: result?.fused_heatmap_url,
      alt: "Fused heatmap",
    },
    {
      key: "masked_oct_url",
      title: "Masked OCT",
      url: result?.masked_oct_url,
      alt: "Masked OCT output",
    },
    {
      key: "result_image_url",
      title: "Combined Validation Figure",
      url: result?.result_image_url,
      alt: "Combined fused analysis result",
    },
  ].filter((item) => item.url)

  const validationEntries = result?.validation
    ? Object.entries(result.validation).filter(([, value]) => typeof value === "number")
    : []

  const bestValidationEntry = validationEntries.reduce(
    (best, current) => (current[1] > best[1] ? current : best),
    validationEntries[0] ?? ["", 0],
  )

  const prettyMetricLabel = (key) => {
    const labels = {
      fused: "fused mask",
      layer2: "layer 2 mask",
      layer3: "layer 3 mask",
      layer4: "layer 4 mask",
    }

    return labels[key] ?? key
  }

  const prettyLayerName = (key) => {
    const labels = {
      layer2: "Layer 2",
      layer3: "Layer 3",
      layer4: "Layer 4",
    }

    return labels[key] ?? key
  }

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
          detected disease label, the uploaded scan, fused Grad-CAM outputs, and validation
          confidence retention from the backend.
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
            {result.predicted_class ?? result.prediction}
          </h2>
        </div>
        <div className="flex flex-wrap gap-4">
          {analysisImages.slice(0, 4).map((item) => (
            <a
              key={item.key}
              href={item.url}
              target="_blank"
              rel="noreferrer"
              className="text-sm font-medium text-cyan-200 transition hover:text-white"
            >
              Open {item.title}
            </a>
          ))}
        </div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        {result.heatmap_interpretation && (
          <article className="rounded-[1.5rem] border border-cyan-300/20 bg-cyan-300/5 p-5">
            <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-300">
              Visual Attention Summary
            </p>
            <h3 className="mt-2 text-xl font-semibold text-white">
              {result.heatmap_interpretation.title}
            </h3>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              {result.heatmap_interpretation.summary}
            </p>
            <div className="mt-4 rounded-xl bg-slate-950/50 p-3">
              <p className="text-xs font-bold uppercase text-slate-500">Model Focus Note</p>
              <p className="mt-1 text-sm text-slate-200">{result.heatmap_interpretation.clinical_note}</p>
            </div>
          </article>
        )}
      </div>

      {result.adaptive_strategy && result.adaptive_weights && (
        <article className="mt-5 rounded-[1.5rem] border border-emerald-300/15 bg-emerald-300/5 p-5">
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-100/80">
            Adaptive Fusion Strategy
          </p>
          <h3 className="mt-2 text-2xl font-semibold text-white">
            Confidence-retention adaptive weighting
          </h3>
          <p className="mt-3 text-sm leading-7 text-slate-200">
            {result.adaptive_strategy.explanation}
          </p>
          <p className="mt-3 text-sm leading-7 text-slate-200">
            For this scan, the dominant contributor is{" "}
            <span className="font-semibold text-white">
              {prettyLayerName(result.adaptive_strategy.dominant_layer)}
            </span>
            , because its masked region retained the most predictive evidence before fusion.
          </p>
          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            {Object.entries(result.adaptive_weights).map(([key, value]) => (
              <div key={key} className="rounded-[1.25rem] border border-white/10 bg-slate-950/70 p-4">
                <p className="text-sm uppercase tracking-[0.2em] text-slate-500">
                  {prettyLayerName(key)} Weight
                </p>
                <p className="mt-2 text-2xl font-semibold text-white">
                  {(value * 100).toFixed(2)}%
                </p>
              </div>
            ))}
          </div>
        </article>
      )}

      <article className="mt-5 rounded-[1.5rem] border border-cyan-300/15 bg-slate-950/70 p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-100/80">
              Confidence Level
            </p>
            <p className="mt-2 text-2xl font-semibold text-white">
              {typeof result?.original_confidence === "number"
                ? `${(result.original_confidence * 100).toFixed(2)}%`
                : result.confidence ?? `${Math.round(confidenceScore * 100)}%`}
            </p>
          </div>
          <p className="max-w-md text-sm leading-6 text-slate-400">
            This shows how strongly the fused backend favored the predicted class for the uploaded
            scan before applying any validation masks.
          </p>
        </div>
        <div className="mt-4 h-3 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-emerald-300 to-lime-300 transition-all duration-700"
            style={{ width: `${confidenceScore * 100}%` }}
          />
        </div>
      </article>

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        {analysisImages.map((item) => (
          <article key={item.key} className="rounded-[1.5rem] border border-white/10 bg-slate-950/70 p-4">
            <div className="flex items-center justify-between gap-4">
              <h3 className="text-lg font-semibold text-white">{item.title}</h3>
              <a
                href={item.url}
                target="_blank"
                rel="noreferrer"
                className="text-sm font-medium text-cyan-200 transition hover:text-white"
              >
                Save / Open
              </a>
            </div>
            <img
              src={item.url}
              alt={item.alt}
              className="mt-3 h-[18rem] w-full rounded-[1.25rem] object-contain lg:h-[20rem]"
            />
          </article>
        ))}
      </div>

      {result.ood_analysis && (
        <article
          className={`mt-6 rounded-[2rem] border-2 p-6 ${
            result.ood_analysis.is_ood
              ? "border-rose-500/60 bg-rose-500/15"
              : "border-emerald-500/60 bg-emerald-500/15"
          }`}
        >
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3">
                <div className={`h-4 w-4 rounded-full ${result.ood_analysis.is_ood ? 'bg-rose-500 animate-pulse shadow-lg shadow-rose-500' : 'bg-emerald-500 shadow-lg shadow-emerald-500'}`} />
                <p
                  className={`text-sm font-bold uppercase tracking-[0.3em] ${
                    result.ood_analysis.is_ood ? "text-rose-300" : "text-emerald-300"
                  }`}
                >
                  Reliability & OOD Assessment
                </p>
              </div>
              
              <h3 className="mt-4 text-4xl font-bold text-white leading-tight">
                {result.ood_analysis.title}
              </h3>
              
              <p className="mt-4 text-lg leading-relaxed text-slate-200">
                {result.ood_analysis.summary}
              </p>

              {result.ood_analysis.reasons?.length > 0 && (
                <div className="mt-5 space-y-3">
                  <p className="text-xs font-bold uppercase text-slate-400 tracking-widest">Detection Insights</p>
                  <ul className="space-y-2">
                    {result.ood_analysis.reasons.map((reason, idx) => (
                      <li key={idx} className="flex items-start gap-3 text-sm text-slate-300 italic">
                        <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-slate-500" />
                        {reason}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className={`mt-8 rounded-2xl border-2 p-5 ${
                result.ood_analysis.is_ood
                  ? "bg-rose-500/20 border-rose-500/40"
                  : "bg-emerald-500/20 border-emerald-500/40"
              }`}>
                <p className={`text-xs font-bold uppercase tracking-[0.2em] ${
                  result.ood_analysis.is_ood ? "text-rose-300" : "text-emerald-300"
                }`}>
                  Clinical Recommendation
                </p>
                <p className="mt-3 text-base leading-7 text-slate-100 font-medium">
                  {result.ood_analysis.recommendation}
                </p>
              </div>
            </div>

            {result.ood_analysis.metrics && (
              <div className="grid w-full shrink-0 gap-4 sm:grid-cols-2 lg:w-80 lg:grid-cols-1">
                <div className={`rounded-2xl border-2 p-5 shadow-lg ${
                  result.ood_analysis.is_ood
                    ? "border-rose-500/40 bg-slate-950/80"
                    : "border-emerald-500/40 bg-slate-950/80"
                }`}>
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      Confidence Margin
                    </p>
                    <span className="text-xs text-slate-400 font-medium">Threshold: 20%</span>
                  </div>
                  <p className="mt-2 text-3xl font-bold text-white">
                    {((result.ood_analysis.metrics.confidence_margin || 0) * 100).toFixed(1)}%
                  </p>
                  <div className="mt-3 h-2 w-full bg-white/10 rounded-full overflow-hidden border border-white/5">
                    <div 
                      className="h-full bg-gradient-to-r from-cyan-400 to-cyan-500 transition-all duration-1000 shadow-lg shadow-cyan-500/50" 
                      style={{ width: `${Math.min(100, (result.ood_analysis.metrics.confidence_margin || 0) * 100)}%` }}
                    />
                  </div>
                </div>

                <div className={`rounded-2xl border-2 p-5 shadow-lg ${
                  result.ood_analysis.is_ood
                    ? "border-rose-500/40 bg-slate-950/80"
                    : "border-emerald-500/40 bg-slate-950/80"
                }`}>
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      Norm. Entropy
                    </p>
                    <span className="text-xs text-slate-400 font-medium">Max: 1.0</span>
                  </div>
                  <p className="mt-2 text-3xl font-bold text-white">
                    {(result.ood_analysis.metrics.normalized_entropy || 0).toFixed(3)}
                  </p>
                  <div className="mt-3 h-2 w-full bg-white/10 rounded-full overflow-hidden border border-white/5">
                    <div 
                      className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 transition-all duration-1000 shadow-lg shadow-emerald-500/50" 
                      style={{ width: `${(result.ood_analysis.metrics.normalized_entropy || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </article>
      )}

      {result.validation && (
        <article className="mt-6 rounded-[1.5rem] border border-cyan-300/15 bg-cyan-300/5 p-5">
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-100/80">
            Confidence Retention Validation
          </p>
          <p className="mt-3 text-sm leading-7 text-slate-300">
            Think of this as a <strong>"Detective Stress Test."</strong> We show the AI only the highlighted
            region and ask: "Can you still make the same diagnosis?" These percentages tell us how
            much of the original "truth" was captured in each specific area.
          </p>
          <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {Object.entries(result.validation).map(([key, value]) => (
              <div key={key} className="rounded-[1.25rem] border border-white/10 bg-slate-950/70 p-4">
                <p className="text-sm uppercase tracking-[0.2em] text-slate-500">
                  {prettyMetricLabel(key)}
                </p>
                <p className="mt-2 text-2xl font-semibold text-white">
                  {typeof value === "number" ? `${(value * 100).toFixed(2)}%` : value}
                </p>
              </div>
            ))}
          </div>

        
        </article>
        )}
        </section>
        )
        }

