import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const topics = [
  {
    title: "Course Context",
    description:
      "CS4063 - Natural Language Processing explores applied AI techniques including data preparation, model evaluation, and deployment strategies.",
  },
  {
    title: "Why Financial Forecasting",
    description:
      "Markets react to structured and unstructured data. Forecasting enables risk anticipation, portfolio management, and early trend discovery.",
  },
  {
    title: "Upcoming ML Integration",
    description:
      "Future versions will include ARIMA, LSTM, and Transformer pipelines with sentiment-aware insights and experiment tracking.",
  },
];

export default function AboutPage() {
  return (
    <div className="space-y-12">
      {/* Header Section */}
      <div className="text-center md:text-left">
        <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">
          About
        </p>
        <h1 className="mt-2 text-4xl font-bold tracking-tight text-foreground">
          FinTech Forecaster – Assignment Brief
        </h1>
        <p className="mt-4 max-w-2xl text-sm text-muted-foreground">
          This project is part of CS4063 - Natural Language Processing. It
          demonstrates a clean Next.js experience, prepared for ML-driven
          forecasts. Submission is due <span className="font-medium">Tuesday, October 7th, 10:00am</span>.
        </p>
      </div>

      {/* Topics Section */}
      <div className="grid gap-6 md:grid-cols-3">
        {topics.map((topic) => (
          <Card
            key={topic.title}
            className="group relative overflow-hidden rounded-2xl border border-border/40 bg-card/80 backdrop-blur shadow-md transition hover:shadow-lg hover:border-primary/50"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
            <CardHeader>
              <CardTitle className="text-lg font-semibold">
                {topic.title}
              </CardTitle>
              <CardDescription className="text-sm leading-relaxed">
                {topic.description}
              </CardDescription>
            </CardHeader>
          </Card>
        ))}
      </div>

      {/* Motivation Section */}
      <Card className="rounded-2xl border border-border/40 bg-card/80 backdrop-blur shadow-md">
        <CardHeader>
          <CardTitle className="text-xl font-semibold">Motivation</CardTitle>
          <CardDescription>
            Accurate forecasting empowers smarter decision-making, hedging
            strategies, and personalized financial insights. Integrating NLP
            with quantitative indicators is a key milestone.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground leading-relaxed">
          <p>
            As you extend this assignment, connect Alpha Vantage data with
            pipeline-ready datasets, explore different model families, and
            benchmark with rolling windows. A rich UI ensures stakeholders stay
            aligned as models iterate in the background.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
