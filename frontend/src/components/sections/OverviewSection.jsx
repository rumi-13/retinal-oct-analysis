import { projectHighlights } from "../../data/siteContent"

export function OverviewSection() {
  return (
    <section id="overview" className="mx-auto max-w-7xl px-6 py-20 lg:px-8">
      <div className="max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-300/80">
          Project Overview
        </p>
        <h2 className="mt-4 font-serif text-4xl font-semibold text-white">
          A clearer interface for an explainable OCT diagnosis workflow.
        </h2>
        <p className="mt-5 text-lg leading-8 text-slate-300">
          The platform is designed to present the project like a professional healthcare AI
          product while keeping the model pipeline untouched. It explains the problem,
          showcases the workflow, and separates the clinical screening action into its own page.
        </p>
      </div>

      <div className="mt-10 grid gap-6 lg:grid-cols-3">
        {projectHighlights.map((item) => (
          <article
            key={item.title}
            className="rounded-[1.75rem] border border-white/10 bg-white/[0.04] p-6 backdrop-blur"
          >
            <p className="text-sm uppercase tracking-[0.2em] text-slate-400">{item.kicker}</p>
            <h3 className="mt-3 text-2xl font-semibold text-white">{item.title}</h3>
            <p className="mt-4 leading-7 text-slate-300">{item.description}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
