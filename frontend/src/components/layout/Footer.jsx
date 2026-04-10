export function Footer() {
  return (
    <footer className="border-t border-white/10 bg-slate-950/80">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-6 py-8 text-sm text-slate-400 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div>
          <p className="font-semibold text-slate-200">Retinal OCT Analysis Platform</p>
          <p>AI-assisted screening experience for OCT image interpretation and visual explainability.</p>
        </div>
        <div className="flex gap-5">
          <a href="#/" className="transition hover:text-white">
            Home
          </a>
          <a href="#/project" className="transition hover:text-white">
            Project
          </a>
          <a href="#/predict" className="transition hover:text-white">
            Prediction
          </a>
        </div>
      </div>
    </footer>
  )
}
