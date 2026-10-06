import ObservatoryApp from "@/components/observatory/ObservatoryApp";

/** The First-Light basis is a static, immutable, integrity-sealed bundle. */
const FIRST_LIGHT_BUNDLE = "/observatory/first-light";

export default function ObservatoryPage() {
  return <ObservatoryApp base={FIRST_LIGHT_BUNDLE} />;
}
