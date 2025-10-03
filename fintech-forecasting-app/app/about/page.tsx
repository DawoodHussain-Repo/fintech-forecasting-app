import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const topics = [
  {
    title: "Course context",
    description:
      "CS4063 - Natural Language Processing focuses on applied AI techniques, including data preparation, model evaluation, and deployment considerations.",
  },
  {
    title: "Why financial forecasting",
    description:
      "Markets react to vast streams of structured and unstructured data. Forecasting helps investors, analysts, and policymakers anticipate risk, manage portfolios, and uncover emerging trends.",
  },
  {
    title: "Upcoming ML integration",
    description:
      "Future iterations will plug in ARIMA, LSTM, and Transformer pipelines for automated predictions, sentiment-aware features, and experiment tracking.",
  },
];

export default function AboutPage() {
  return (
    <div className="space-y-10">
      <div>
        <p className="text-sm uppercase tracking-[0.3em] text-muted-foreground">
          About
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-foreground">
          FinTech Forecaster – Assignment brief
        </h1>
        <p className="mt-4 max-w-2xl text-sm text-muted-foreground">
          This project is part of CS4063 - Natural Language Processing. The
          deliverable showcases a clean Next.js experience ready for integrating
          ML-driven forecasts. Submission is due Tuesday, October 7th, 10:00am.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {topics.map((topic) => (
          <Card key={topic.title} className="border-none bg-card/80 shadow-sm">
            <CardHeader>
              <CardTitle className="text-xl">{topic.title}</CardTitle>
              <CardDescription>{topic.description}</CardDescription>
            </CardHeader>
          </Card>
        ))}
      </div>

      <Card className="border-none bg-card/80">
        <CardHeader>
          <CardTitle>Motivation</CardTitle>
          <CardDescription>
            Accurate forecasting empowers smarter decision-making, hedging
            strategies, and personalized financial insights. Combining NLP
            signals with quantitative indicators is a key milestone for the
            final project.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          <p>
            As you extend this assignment, connect Alpha Vantage data with
            pipeline-friendly datasets, experiment with different model
            families, and evaluate performance using rolling windows. Rich UI
            feedback keeps stakeholders aligned while models iterate in the
            background.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
