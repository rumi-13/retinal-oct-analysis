import { useEffect, useState } from "react"

import { Footer } from "../components/layout/Footer"
import { Header } from "../components/layout/Header"
import { HomePage } from "../pages/HomePage"
import { PredictionPage } from "../pages/PredictionPage"
import { ProjectPage } from "../pages/ProjectPage"

const getRouteFromHash = () => {
  const hash = window.location.hash.replace("#", "")

  if (hash === "/predict") {
    return "predict"
  }

  if (hash === "/project") {
    return "project"
  }

  return "home"
}

function App() {
  const [route, setRoute] = useState(getRouteFromHash)

  useEffect(() => {
    const handleHashChange = () => {
      setRoute(getRouteFromHash())
      window.scrollTo({ top: 0, behavior: "smooth" })
    }

    window.addEventListener("hashchange", handleHashChange)

    return () => {
      window.removeEventListener("hashchange", handleHashChange)
    }
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="relative overflow-x-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(34,197,94,0.18),_transparent_30%),radial-gradient(circle_at_top_right,_rgba(14,165,233,0.22),_transparent_35%),linear-gradient(180deg,_rgba(15,23,42,0.95),_rgba(2,6,23,1))]" />
        <div className="absolute inset-x-0 top-0 h-px bg-white/10" />

        <div className="relative">
          <Header currentRoute={route} />
          <main>
            {route === "predict" ? (
              <PredictionPage />
            ) : route === "project" ? (
              <ProjectPage />
            ) : (
              <HomePage />
            )}
          </main>
          <Footer />
        </div>
      </div>
    </div>
  )
}

export default App
