import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface BaseProps {
  className?: string;
  children?: ReactNode;
}

export function Card({ className, children }: BaseProps) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-border/60 bg-card/80 backdrop-blur-md shadow-lg shadow-black/5 dark:shadow-none",
        className
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({ className, children }: BaseProps) {
  return (
    <div className={cn("flex flex-col space-y-1.5 p-6", className)}>
      {children}
    </div>
  );
}

export function CardTitle({ className, children }: BaseProps) {
  return (
    <h3
      className={cn(
        "text-xl font-semibold leading-tight tracking-tight",
        className
      )}
    >
      {children}
    </h3>
  );
}

export function CardDescription({ className, children }: BaseProps) {
  return (
    <p className={cn("text-sm text-muted-foreground", className)}>{children}</p>
  );
}

export function CardContent({ className, children }: BaseProps) {
  return <div className={cn("p-6 pt-0", className)}>{children}</div>;
}

export function CardFooter({ className, children }: BaseProps) {
  return (
    <div className={cn("flex items-center p-6 pt-0", className)}>
      {children}
    </div>
  );
}
