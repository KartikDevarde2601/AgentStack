import type { ThemeOption } from '@openai/chatkit-react';

export const CHATKIT_FOODWORKS_THEME = {
  colorScheme: 'light',
  radius: 'pill',
  density: 'normal',
  typography: {
    baseSize: 17,
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    fontSources: [
      {
        family: 'Inter',
        src: 'https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7W0Q5nw.woff2',
        weight: '400 700',
        display: 'swap',
      },
    ],
  },
  color: {
    grayscale: {
      hue: 50,
      tint: 2,
      shade: -1,
    },
    accent: {
      primary: '#583AD8',
      level: 2,
    },
    surface: {
      background: '#F4F3EC',
      foreground: '#FFFFFF',
    },
  },
} as const satisfies ThemeOption;
