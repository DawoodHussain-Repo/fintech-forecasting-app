import type { InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  variant?: "default" | "ghost";
}

export function Input({
  className,
  variant = "default",
  ...props
}: InputProps) {
  return (
    <input
      className={cn(
        "h-11 w-full rounded-xl border border-border/60 bg-background/80 px-4 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-70",
        variant === "ghost" &&
          "bg-transparent border-transparent focus-visible:ring-0 focus-visible:ring-offset-0",
        className
      )}
      {...props}
    />
  );
}
