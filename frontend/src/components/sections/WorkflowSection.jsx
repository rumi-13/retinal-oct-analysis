import { workflowSteps } from "../../data/siteContent"

export function WorkflowSection() {
  return (
    <section className="mx-auto max-w-7xl px-6 py-20 lg:px-8">
      <div className="grid gap-8 lg:grid-cols-[0.75fr_1.25fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-300/80">
            Model Journey
          </p>
          <h2 className="mt-4 font-serif text-4xl font-semibold text-white">
            From uploaded image to interpretable output.
          </h2>
          <p className="mt-5 text-lg leading-8 text-slate-300">
            The redesigned site clarifies the exact screening path so users immediately
            understand what the tool does before they interact with the prediction page.
          </p>
        </div>

        <div className="grid gap-5">
          {workflowSteps.map((step) => (
            <div
              key={step.number}
              className="flex gap-5 rounded-[1.75rem] border border-white/10 bg-white/[0.04] p-6"
            >
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-white text-lg font-semibold text-slate-950">
                {step.number}
              </div>
              <div>
                <h3 className="text-2xl font-semibold text-white">{step.title}</h3>
                <p className="mt-3 leading-7 text-slate-300">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
