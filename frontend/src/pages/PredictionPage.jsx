import { useState } from "react"

import { ResultPanel } from "../components/prediction/ResultPanel"
import { UploadPanel } from "../components/prediction/UploadPanel"

export function PredictionPage() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileChange = (event) => {
    setFile(event.target.files[0])
  }

  const handleSubmit = async (event) => {
    event.preventDefault()

    if (!file) {
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        let message = "Prediction request failed."

        try {
          const errorData = await response.json()
          if (errorData?.error) {
            message = errorData.error
          }
        } catch {
          message = `Prediction request failed with status ${response.status}.`
        }

        throw new Error(message)
      }

      const data = await response.json()
      setResult(data)
    } catch (submitError) {
      setError(submitError.message || "Failed to analyze image. Ensure Flask is running.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-6 py-6 lg:px-8 lg:py-8">
      <div className="grid gap-5">
        <section className="rounded-[1.75rem] border border-amber-300/25 bg-amber-300/10 p-5 backdrop-blur">
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-amber-100/90">
            Important Upload Warning
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-white">
            Please upload genuine retinal OCT images only.
          </h2>
          <p className="mt-3 max-w-4xl text-sm leading-7 text-amber-50/90 sm:text-base">
            This model is trained for retinal OCT scan classification. If you upload random images
            such as logos, buildings, documents, or other non-OCT pictures, the system may still
            return a disease label, but that result can be incorrect and medically meaningless.
          </p>
        </section>

        <UploadPanel
          file={file}
          loading={loading}
          onFileChange={handleFileChange}
          onSubmit={handleSubmit}
        />
        <ResultPanel result={result} loading={loading} error={error} />
      </div>
    </div>
  )
}
