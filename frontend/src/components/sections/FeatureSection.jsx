import { featureCards } from "../../data/siteContent"

export function FeatureSection() {
  return (
    <section className="mx-auto max-w-7xl px-6 py-8 lg:px-8 lg:py-12">
      <div className="rounded-[2rem] border border-white/10 bg-slate-900/65 p-8 backdrop-blur">
        <div className="max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-cyan-200/80">
            Key Features
          </p>
          <h2 className="mt-4 font-serif text-4xl font-semibold text-white">
            What this web experience communicates.
          </h2>
        </div>

        <div className="mt-10 grid gap-6 md:grid-cols-2 xl:grid-cols-4">
          {featureCards.map((card) => (
            <div
              key={card.title}
              className="rounded-[1.5rem] border border-white/10 bg-slate-950/70 p-5"
            >
              <p className="text-sm uppercase tracking-[0.24em] text-slate-500">{card.tag}</p>
              <h3 className="mt-4 text-xl font-semibold text-white">{card.title}</h3>
              <p className="mt-3 text-sm leading-7 text-slate-300">{card.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
