// Type declarations for lucide-react
declare module "lucide-react" {
  import { FC, SVGProps } from "react";

  export interface IconProps extends SVGProps<SVGSVGElement> {
    size?: number | string;
    color?: string;
    strokeWidth?: number | string;
    absoluteStrokeWidth?: boolean;
  }

  export type Icon = FC<IconProps>;

  // Common icons used in the app
  export const TrendingUp: Icon;
  export const TrendingDown: Icon;
  export const Activity: Icon;
  export const BarChart3: Icon;
  export const LineChart: Icon;
  export const PieChart: Icon;
  export const Calendar: Icon;
  export const Clock: Icon;
  export const DollarSign: Icon;
  export const Percent: Icon;
  export const Target: Icon;
  export const Zap: Icon;
  export const Settings: Icon;
  export const User: Icon;
  export const Menu: Icon;
  export const X: Icon;
  export const ChevronDown: Icon;
  export const ChevronUp: Icon;
  export const ChevronLeft: Icon;
  export const ChevronRight: Icon;
  export const Plus: Icon;
  export const Minus: Icon;
  export const Search: Icon;
  export const Filter: Icon;
  export const Download: Icon;
  export const Upload: Icon;
  export const RefreshCw: Icon;
  export const Eye: Icon;
  export const EyeOff: Icon;
  export const Star: Icon;
  export const Heart: Icon;
  export const Home: Icon;
  export const Globe: Icon;
  export const Mail: Icon;
  export const Phone: Icon;
  export const MapPin: Icon;
  export const Calendar: Icon;
  export const Clock: Icon;
  export const Users: Icon;
  export const Building: Icon;
  export const Briefcase: Icon;
  export const CreditCard: Icon;
  export const Shield: Icon;
  export const Lock: Icon;
  export const Unlock: Icon;
  export const Key: Icon;
  export const Info: Icon;
  export const AlertCircle: Icon;
  export const CheckCircle: Icon;
  export const XCircle: Icon;
  export const HelpCircle: Icon;
  export const Loader: Icon;
  export const Spinner: Icon;
}
