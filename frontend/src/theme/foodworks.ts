import type { ThemeOption } from '@openai/chatkit-react';

export const CHATKIT_FOODWORKS_THEME = {
  colorScheme: 'light',
  radius: 'pill',
  density: 'normal',
  typography: {
    baseSize: 15,
    fontFamily: "'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    fontSources: [
      {
        family: 'Roboto',
        src: 'https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMa3yUBHMdazQ.woff2',
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
