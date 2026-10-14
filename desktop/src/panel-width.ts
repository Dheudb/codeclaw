export const DEFAULT_PANEL_WIDTH = 318;
export function panelLimits(viewport: number) {
  return { min: 280, max: Math.max(280, Math.min(760, viewport - (viewport <= 1200 ? 220 : 244) - 400)) };
}
export function clampPanelWidth(width: number, viewport: number) {
  const { min, max } = panelLimits(viewport);
  return Math.round(Math.max(min, Math.min(max, Number.isFinite(width) ? width : DEFAULT_PANEL_WIDTH)));
}
