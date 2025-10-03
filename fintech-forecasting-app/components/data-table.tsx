import { Sparklines, SparklinesLine } from "react-sparklines";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { formatChangePercent, formatCurrency } from "@/lib/utils";

export interface WatchlistRow {
  symbol: string;
  price: number;
  changePercent: number;
  sparkline: number[];
  currency?: string;
}

interface DataTableProps {
  items: WatchlistRow[];
  onRemove?: (symbol: string) => void;
}

export function DataTable({ items, onRemove }: DataTableProps) {
  if (items.length === 0) {
    return (
      <div className="flex min-h-[240px] flex-col items-center justify-center rounded-2xl border border-dashed border-border/60 bg-card/60 text-center">
        <p className="text-lg font-semibold text-foreground">No items yet</p>
        <p className="max-w-sm text-sm text-muted-foreground">
          No items yet – search and add from Dashboard.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-border/60 bg-card/80 shadow">
      <table className="min-w-full divide-y divide-border/60 text-sm">
        <thead className="bg-background/70">
          <tr className="text-left uppercase tracking-wide text-xs text-muted-foreground">
            <th className="px-6 py-4">Symbol</th>
            <th className="px-6 py-4">Current Price</th>
            <th className="px-6 py-4">Daily % Change</th>
            <th className="px-6 py-4">Trend</th>
            <th className="px-6 py-4 text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/60 bg-background/40">
          {items.map((item) => {
            const intent =
              item.changePercent > 0
                ? "success"
                : item.changePercent < 0
                ? "danger"
                : "default";
            return (
              <tr key={item.symbol} className="transition hover:bg-muted/40">
                <td className="px-6 py-4 font-semibold">{item.symbol}</td>
                <td className="px-6 py-4">
                  {formatCurrency(item.price, item.currency ?? "USD")}
                </td>
                <td className="px-6 py-4">
                  <Badge intent={intent}>
                    {formatChangePercent(item.changePercent)}
                  </Badge>
                </td>
                <td className="px-6 py-4">
                  <Sparklines
                    data={item.sparkline}
                    svgWidth={120}
                    svgHeight={36}
                  >
                    <SparklinesLine
                      style={{
                        fill: "none",
                        strokeWidth: 2,
                        stroke: item.changePercent >= 0 ? "#34d399" : "#f87171",
                      }}
                    />
                  </Sparklines>
                </td>
                <td className="px-6 py-4 text-right">
                  <Button
                    variant="ghost"
                    className="text-xs text-muted-foreground hover:text-foreground"
                    onClick={() => onRemove?.(item.symbol)}
                  >
                    Remove
                  </Button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
