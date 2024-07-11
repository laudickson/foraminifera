module.exports = {
    root: true,
    env: { browser: true, es2020: true },
    extends: [
      'eslint:recommended',
      'plugin:@typescript-eslint/recommended',
      'plugin:react-hooks/recommended',
      "plugin:prettier/recommended"
    ],
    ignorePatterns: ['dist', '.eslintrc.cjs'],
    parser: '@typescript-eslint/parser',
    plugins: ['react-refresh'],
    rules: {
      "max-len": ["error", { "code": 120 }],
      "indent": "off",
      "@typescript-eslint/indent": "off",
      "@typescript-eslint/semi": "off", // use semi-rule above
      "@typescript-eslint/strict-boolean-expressions": "off", // too strict out of box
      "@typescript-eslint/member-delimiter-style": ["error", {
          "multiline": {
              "delimiter": "semi",
              "requireLast": true
          }
      }],
      "@typescript-eslint/object-curly-spacing": ["error", "always"],
      "@typescript-eslint/no-explicit-any": ["error", { "ignoreRestArgs": true }],
      "react-hooks/exhaustive-deps": "off"
    },
};
