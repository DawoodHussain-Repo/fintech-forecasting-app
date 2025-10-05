"use client";

import { useState, useRef, useEffect } from "react";
import { gsap } from "gsap";
import { Search, X } from "lucide-react";

interface NeonSearchBarProps {
  placeholder?: string;
  onSearch: (value: string) => void;
  suggestions?: string[];
  loading?: boolean;
}

export function NeonSearchBar({
  placeholder = "Search stocks, crypto, forex...",
  onSearch,
  suggestions = [],
  loading = false,
}: NeonSearchBarProps) {
  const [value, setValue] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (glowRef.current) {
      gsap.to(glowRef.current, {
        opacity: isFocused ? 1 : 0,
        scale: isFocused ? 1 : 0.95,
        duration: 0.3,
        ease: "power2.out",
      });
    }
  }, [isFocused]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) {
      onSearch(value.trim());
      setShowSuggestions(false);
    }
  };

  const handleClear = () => {
    setValue("");
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  const filteredSuggestions = suggestions.filter((s) =>
    s.toLowerCase().includes(value.toLowerCase())
  );

  return (
    <div ref={containerRef} className="relative w-full max-w-2xl mx-auto">
      {/* Neon glow effect */}
      <div
        ref={glowRef}
        className="absolute inset-0 bg-primary/20 blur-xl rounded-full opacity-0"
        style={{ transform: "scale(0.95)" }}
      />

      {/* Search container */}
      <form onSubmit={handleSubmit} className="relative">
        <div
          className={`glass relative overflow-hidden transition-all duration-300 ${
            isFocused ? "ring-2 ring-primary/50" : ""
          }`}
        >
          {/* Animated background gradient */}
          <div
            className={`absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/10 to-primary/0 transition-opacity duration-300 ${
              isFocused ? "opacity-100" : "opacity-0"
            }`}
          />

          <div className="relative flex items-center px-6 py-4">
            {/* Search icon */}
            <Search
              className={`w-5 h-5 mr-3 transition-all duration-300 ${
                isFocused ? "text-primary glow-sm" : "text-muted-foreground"
              }`}
            />

            {/* Input */}
            <input
              ref={inputRef}
              type="text"
              value={value}
              onChange={(e) => {
                setValue(e.target.value);
                setShowSuggestions(true);
              }}
              onFocus={() => {
                setIsFocused(true);
                if (value) setShowSuggestions(true);
              }}
              onBlur={() => {
                setIsFocused(false);
                setTimeout(() => setShowSuggestions(false), 200);
              }}
              placeholder={placeholder}
              className="flex-1 bg-transparent outline-none text-foreground placeholder:text-muted-foreground text-lg"
            />

            {/* Loading spinner */}
            {loading && (
              <div className="w-5 h-5 mr-3 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
            )}

            {/* Clear button */}
            {value && !loading && (
              <button
                type="button"
                onClick={handleClear}
                className="p-1 rounded-full hover:bg-primary/10 transition-colors"
              >
                <X className="w-4 h-4 text-muted-foreground hover:text-primary" />
              </button>
            )}
          </div>

          {/* Bottom neon line */}
          <div
            className={`absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent transition-all duration-300 ${
              isFocused ? "w-full opacity-100" : "w-0 opacity-0"
            }`}
          />
        </div>
      </form>

      {/* Suggestions dropdown */}
      {showSuggestions && filteredSuggestions.length > 0 && value && (
        <div className="absolute top-full mt-2 w-full glass z-50 overflow-hidden">
          <div className="p-2">
            {filteredSuggestions.slice(0, 5).map((suggestion, index) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => {
                  setValue(suggestion);
                  onSearch(suggestion);
                  setShowSuggestions(false);
                }}
                className="w-full text-left px-4 py-3 rounded-lg hover:bg-primary/10 transition-colors group"
                style={{
                  animation: `slideDown 0.3s ease-out ${index * 0.05}s both`,
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-foreground font-medium group-hover:text-primary transition-colors">
                    {suggestion}
                  </span>
                  <Search className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
