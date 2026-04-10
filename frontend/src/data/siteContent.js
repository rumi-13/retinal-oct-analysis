export const projectHighlights = [
  {
    kicker: "Purpose",
    title: "AI-assisted retinal screening",
    description:
      "The project uses a trained deep learning model to classify retinal OCT images into clinically relevant categories and present the result through a cleaner web interface.",
  },
  {
    kicker: "Explainability",
    title: "Visual reasoning support",
    description:
      "Grad-CAM heatmaps help users understand where the network focused during inference, making the result more interpretable than a plain label alone.",
  },
  {
    kicker: "Product Framing",
    title: "Professional presentation",
    description:
      "The redesigned site introduces a real homepage, clear content hierarchy, and a dedicated prediction workspace so the project feels like a polished portfolio product.",
  },
]

export const featureCards = [
  {
    tag: "Model",
    title: "TensorFlow inference",
    description:
      "The backend loads a saved Keras model once and preprocesses each retinal image consistently before prediction.",
  },
  {
    tag: "Output",
    title: "Disease name focus",
    description:
      "The prediction page emphasizes the disease label and supporting visual outputs, keeping the UI focused on the most useful information.",
  },
  {
    tag: "UX",
    title: "Separated journeys",
    description:
      "Visitors first learn what the project is, then move into a dedicated analysis workflow instead of landing directly on a technical upload form.",
  },
  {
    tag: "Structure",
    title: "Maintainable folders",
    description:
      "Pages, sections, layout elements, and prediction components are split into clear folders so future UI work stays manageable.",
  },
]

export const workflowSteps = [
  {
    number: "01",
    title: "Image intake",
    description:
      "A retinal OCT image is uploaded through the prediction page and sent to the existing Flask endpoint using the same form submission flow.",
  },
  {
    number: "02",
    title: "Model inference",
    description:
      "The backend resizes and preprocesses the image, performs deep learning inference, and selects the disease class with the highest model score.",
  },
  {
    number: "03",
    title: "Explainable result",
    description:
      "The UI presents the predicted disease name alongside the original OCT image and its Grad-CAM heatmap for visual interpretation.",
  },
]

export const diseaseDetails = [
  {
    id: "CNV",
    accent: "bg-emerald-300/10 text-emerald-100 border-emerald-300/20",
    title: "CNV",
    name: "Choroidal Neovascularization",
    description:
      "CNV refers to abnormal blood vessel growth beneath the retina. In retinal imaging workflows, it is important because these vessels can leak fluid or blood and threaten central vision.",
  },
  {
    id: "DME",
    accent: "bg-cyan-300/10 text-cyan-100 border-cyan-300/20",
    title: "DME",
    name: "Diabetic Macular Edema",
    description:
      "DME is swelling in the macula caused by leakage from damaged retinal blood vessels, commonly associated with diabetic retinopathy and reduced visual clarity.",
  },
  {
    id: "DRUSEN",
    accent: "bg-amber-300/10 text-amber-100 border-amber-300/20",
    title: "DRUSEN",
    name: "Drusen Deposits",
    description:
      "Drusen are yellow extracellular deposits beneath the retina. Their presence is commonly discussed in relation to age-related macular degeneration assessment.",
  },
  {
    id: "NORMAL",
    accent: "bg-violet-300/10 text-violet-100 border-violet-300/20",
    title: "NORMAL",
    name: "No major abnormality detected",
    description:
      "NORMAL indicates the scan most closely matches the non-pathological class learned during training, meaning no major disease pattern was highlighted by the model.",
  },
]
