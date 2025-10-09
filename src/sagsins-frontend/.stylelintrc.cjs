module.exports = {
  extends: [
    "stylelint-config-standard",
  ],
  rules: {
    // Allow Tailwind's at-rules so editors and stylelint won't flag them as unknown
    "at-rule-no-unknown": [
      true,
      {
        "ignoreAtRules": [
          "tailwind",
          "apply",
          "variants",
          "responsive",
          "screen"
        ]
      }
    ]
  }
};
