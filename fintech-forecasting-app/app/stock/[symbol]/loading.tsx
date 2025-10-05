import { Loader } from "lucide-react";

export default function Loading() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center space-y-4">
        <Loader className="h-16 w-16 animate-spin text-primary mx-auto" />
        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-primary">
            Loading Stock Data
          </h2>
          <p className="text-muted-foreground">
            Fetching latest market information...
          </p>
        </div>

        {/* Skeleton loaders */}
        <div className="max-w-4xl mx-auto mt-8 space-y-4">
          <div className="glass rounded-lg p-6 animate-pulse">
            <div className="h-8 bg-primary/20 rounded w-1/3 mb-4"></div>
            <div className="h-12 bg-primary/20 rounded w-1/2"></div>
          </div>
          <div className="glass rounded-lg p-6 animate-pulse">
            <div className="h-64 bg-primary/10 rounded"></div>
          </div>
        </div>
      </div>
    </div>
  );
}
