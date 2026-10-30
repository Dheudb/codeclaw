export const DEFAULT_PANEL_WIDTH = 318;
export const DEFAULT_SIDEBAR_WIDTH = 244;
export function defaultSidebarWidth(viewport: number) {
  return viewport <= 1200 ? 220 : DEFAULT_SIDEBAR_WIDTH;
}
export function sidebarLimits(viewport: number, panelOpen = true) {
  return { min: 200, max: Math.max(200, Math.min(480, viewport - 400 - (panelOpen ? 280 : 0))) };
}
export function clampSidebarWidth(width: number, viewport: number, panelOpen = true) {
  const { min, max } = sidebarLimits(viewport, panelOpen);
  return Math.round(Math.max(min, Math.min(max, Number.isFinite(width) ? width : defaultSidebarWidth(viewport))));
}
export function panelLimits(viewport: number, sidebarWidth = defaultSidebarWidth(viewport)) {
  return { min: 280, max: Math.max(280, Math.min(12000, viewport - sidebarWidth - 400)) };
}
export function clampPanelWidth(width: number, viewport: number, sidebarWidth = defaultSidebarWidth(viewport)) {
  const { min, max } = panelLimits(viewport, sidebarWidth);
  return Math.round(Math.max(min, Math.min(max, Number.isFinite(width) ? width : DEFAULT_PANEL_WIDTH)));
}
export function reviewTreeLimits(panelWidth: number) {
  return { min: 120, max: Math.max(120, panelWidth - 180) };
}
export function clampReviewTreeWidth(width: number, panelWidth: number) {
  const { min, max } = reviewTreeLimits(panelWidth);
  return Math.round(Math.max(min, Math.min(max, Number.isFinite(width) ? width : 220)));
}
