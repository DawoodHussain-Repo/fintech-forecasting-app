import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface BadgeProps {
  children?: ReactNode;
  className?: string;
  intent?: "default" | "success" | "danger" | "warning";
}

export function Badge({ className, children, intent = "default" }: BadgeProps) {
  const intentClass = {
    default: "bg-muted/60 text-muted-foreground",
    success: "bg-emerald-500/15 text-emerald-500",
    danger: "bg-red-500/15 text-red-500",
    warning: "bg-amber-500/15 text-amber-500",
  }[intent];

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium uppercase tracking-wide",
        intentClass,
        className
      )}
    >
      {children}
    </span>
  );
}
