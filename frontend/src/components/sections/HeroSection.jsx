import { useState } from "react"

import { diseaseDetails } from "../../data/siteContent"

export function HeroSection() {
  const [selectedDisease, setSelectedDisease] = useState(diseaseDetails[0])

  return (
    <section className="relative">
      <div className="mx-auto grid max-w-7xl gap-6 px-6 py-5 sm:py-6 lg:grid-cols-[minmax(0,0.88fr)_minmax(520px,1.12fr)] lg:items-center lg:px-8 lg:py-8 xl:gap-8">
        <div className="min-w-0">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/25 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100">
            <span className="h-2 w-2 rounded-full bg-cyan-300" />
            Ophthalmic AI Screening Interface
          </div>

          <h1 className="mt-4 max-w-3xl font-serif text-4xl font-semibold leading-[0.96] text-white sm:text-[3.6rem] lg:text-[3.9rem] xl:text-[4.15rem]">
            Turn retinal OCT scans into explainable clinical insights.
          </h1>

          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg sm:leading-7">
            This project combines deep learning, retinal OCT image analysis, and Grad-CAM
            visualization to support fast, intelligible screening for key macular conditions.
          </p>
        </div>

        <div className="relative w-full max-w-4xl justify-self-center lg:max-w-none">
          <div className="absolute -inset-6 rounded-[2rem] bg-gradient-to-r from-emerald-400/20 via-cyan-400/10 to-blue-500/20 blur-3xl" />
          <div className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/5 p-4 shadow-2xl shadow-cyan-950/30 backdrop-blur sm:p-5">
            <div className="grid gap-4">
              <div className="rounded-3xl border border-white/10 bg-slate-900/80 p-4 sm:p-5">
                <p className="text-sm uppercase tracking-[0.22em] text-emerald-300/80">
                  Supported Classes
                </p>
                <div className="mt-3 grid gap-3 xl:grid-cols-[0.8fr_1.2fr]">
                  <div className="flex flex-wrap content-start gap-3 text-sm">
                    {diseaseDetails.map((disease) => {
                      const isActive = selectedDisease.id === disease.id

                      return (
                        <button
                          key={disease.id}
                          type="button"
                          onClick={() => setSelectedDisease(disease)}
                          className={`rounded-full border px-4 py-2 transition ${
                            isActive
                              ? disease.accent
                              : "border-white/10 bg-white/5 text-slate-200 hover:bg-white/10"
                          }`}
                        >
                          {disease.title}
                        </button>
                      )
                    })}
                  </div>

                  <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-500">
                      Selected Condition
                    </p>
                    <h3 className="mt-2 text-lg font-semibold text-white sm:text-xl">
                      {selectedDisease.name}
                    </h3>
                    <p className="mt-2 text-sm leading-6 text-slate-300">
                      {selectedDisease.description}
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-4">
                  <p className="text-3xl font-semibold text-white sm:text-[2rem]">224 x 224</p>
                  <p className="mt-2 text-sm text-slate-300">
                    Standardized image preprocessing for model inference.
                  </p>
                </div>
                <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-4">
                  <p className="text-3xl font-semibold text-white sm:text-[2rem]">Grad-CAM</p>
                  <p className="mt-2 text-sm text-slate-300">
                    Visual attention mapping to show the regions that influenced the model.
                  </p>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900 via-slate-900 to-cyan-950/70 p-4">
                <p className="text-sm uppercase tracking-[0.22em] text-slate-400">
                  Screening Workflow
                </p>
                <div className="mt-3 grid gap-3 sm:grid-cols-3">
                  <div>
                    <p className="text-xl font-semibold text-white">01</p>
                    <p className="mt-1 text-sm text-slate-300">Upload retinal OCT scan.</p>
                  </div>
                  <div>
                    <p className="text-xl font-semibold text-white">02</p>
                    <p className="mt-1 text-sm text-slate-300">Run deep learning inference.</p>
                  </div>
                  <div>
                    <p className="text-xl font-semibold text-white">03</p>
                    <p className="mt-1 text-sm text-slate-300">Review disease label and heatmap.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
