export function CtaSection() {
  return (
    <section className="mx-auto max-w-7xl px-6 py-8 pb-20 lg:px-8">
      <div className="overflow-hidden rounded-[2rem] border border-cyan-300/20 bg-gradient-to-r from-emerald-400/15 via-cyan-400/10 to-blue-500/15 p-8">
        <div className="max-w-3xl">
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-100/80">
            Ready To Analyze
          </p>
          <h2 className="mt-4 font-serif text-4xl font-semibold text-white">
            Open the screening workspace and test an OCT image.
          </h2>
          <p className="mt-4 text-lg leading-8 text-slate-200">
            The prediction workflow remains the same: upload one OCT image, send it to the Flask
            backend, and review the disease label with the Grad-CAM heatmap.
          </p>
          <a
            href="#/predict"
            className="mt-8 inline-flex rounded-full bg-white px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-slate-200"
          >
            Go To Prediction Page
          </a>
        </div>
      </div>
    </section>
  )
}
