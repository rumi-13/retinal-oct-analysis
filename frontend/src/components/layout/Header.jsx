const navItems = [
  { href: "#/", label: "Home" },
  { href: "#/project", label: "Project" },
  { href: "#/predict", label: "Prediction" },
]

export function Header({ currentRoute }) {
  return (
    <header className="sticky top-0 z-30 border-b border-white/10 bg-slate-950/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3 lg:px-8">
        <a href="#/" className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-emerald-400/30 bg-emerald-400/10 text-sm font-semibold text-emerald-200">
            OCT
          </div>
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-300/80">
              Clinical AI
            </p>
            <p className="text-base font-semibold text-white">
              Retinal OCT Analysis
            </p>
          </div>
        </a>

        <nav className="hidden items-center gap-2 md:flex">
          {navItems.map((item) => {
            const isActive =
              (item.href === "#/" && currentRoute === "home") ||
              (item.href === "#/project" && currentRoute === "project") ||
              (item.href === "#/predict" && currentRoute === "predict")

            return (
              <a
                key={item.href}
                href={item.href}
                className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                  isActive
                    ? "bg-white text-slate-950"
                    : "text-slate-300 hover:bg-white/10 hover:text-white"
                }`}
              >
                {item.label}
              </a>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
