module.exports = {
  content: [
    './src/**/*.njk',
    './src/**/*.js',
    './src/**/*.svg',
    './src/**/*.md',
  ],
  plugins: [require('@tailwindcss/forms')],
  theme: {
    extend: {
      fontFamily: {
        jetbrains: '"JetBrains Mono", monospace;',
      },
      colors: {
        'robin_egg_blue': {
          DEFAULT: '#55dde0',
          100: '#0a3435',
          200: '#136869', 
          300: '#1d9c9e',
          400: '#27cfd2',
          500: '#55dde0',
          600: '#79e5e6',
          700: '#9aebed',
          800: '#bcf2f3',
          900: '#ddf8f9'
        },
        'hunyadi_yellow': {
          DEFAULT: '#f6ae2d',
          100: '#382502',
          200: '#704a05',
          300: '#a76f07', 
          400: '#df9409',
          500: '#f6ae2d',
          600: '#f8bf57',
          700: '#facf81',
          800: '#fbdfab',
          900: '#fdefd5'
        },
        'charcoal': {
          DEFAULT: '#2f4858',
          100: '#090e11',
          200: '#131c22',
          300: '#1c2b34',
          400: '#253945', 
          500: '#2f4858',
          600: '#496f87',
          700: '#6c96b0',
          800: '#9db9ca',
          900: '#cedce5'
        },
        'lapis_lazuli': {
          DEFAULT: '#33658a',
          100: '#0a141c',
          200: '#142837',
          300: '#1f3c53',
          400: '#29506e',
          500: '#33658a',
          600: '#4486b9',
          700: '#72a5cb',
          800: '#a1c3dc',
          900: '#d0e1ee'
        },
        'orange_(pantone)': {
          DEFAULT: '#f26419',
          100: '#321303',
          200: '#642706',
          300: '#963a09',
          400: '#c84e0c',
          500: '#f26419',
          600: '#f48346',
          700: '#f7a274',
          800: '#fac1a2',
          900: '#fce0d1'
        }
      },
    },
  },
};
