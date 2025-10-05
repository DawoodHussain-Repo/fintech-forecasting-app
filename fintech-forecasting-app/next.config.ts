import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  env: {
    FINNHUB_API_KEY: process.env.FINNHUB_API_KEY,
  },
};

export default nextConfig;
