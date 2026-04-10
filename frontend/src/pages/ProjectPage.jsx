import { FeatureSection } from "../components/sections/FeatureSection"
import { OverviewSection } from "../components/sections/OverviewSection"

export function ProjectPage() {
  return (
    <div className="py-8 lg:py-10">
      <OverviewSection />
      <FeatureSection />
    </div>
  )
}
