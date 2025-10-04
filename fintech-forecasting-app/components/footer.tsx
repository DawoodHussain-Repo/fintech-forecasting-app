import Link from "next/link";

export function Footer() {
  return (
    <footer className="glass mt-12 border-t border-border/60 py-8 px-4 shadow-inner backdrop-blur-lg">
      <div className="mx-auto flex max-w-7xl flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="h-8 w-8 rounded-full gradient-main flex items-center justify-center text-white font-bold text-lg shadow-md">
            FF
          </span>
          <span className="text-base font-semibold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            FinTech Forecaster
          </span>
        </div>
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <a
            href="https://github.com/your-repo"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-primary"
          >
            GitHub
          </a>
          <a href="/about" className="transition-colors hover:text-primary">
            About
          </a>
          <span className="hidden md:inline">|</span>
          <span className="text-xs">
            &copy; {new Date().getFullYear()} FinTech Forecaster
          </span>
        </div>
      </div>
    </footer>
  );
}
