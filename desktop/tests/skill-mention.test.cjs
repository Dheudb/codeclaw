const test = require("node:test");
const assert = require("node:assert/strict");
test("skill mentions track the caret and preserve surrounding task text", async () => {
  const { skillMention, removeSkillMention } = await import("../src/skill-mention.ts");
  assert.deepEqual(skillMention("$", 1), { start: 0, end: 1, query: "" });
  const task = "$code-review 请检查登录逻辑";
  const mention = skillMention(task, 4);
  assert.equal(mention.query, "code-review");
  assert.equal(removeSkillMention(task, mention), "请检查登录逻辑");
  assert.equal(removeSkillMention("先检查 $debug-issue 这个报错", skillMention("先检查 $debug-issue 这个报错", 16)), "先检查 这个报错");
  assert.equal(skillMention("$联网", 3).query, "联网");
  for (const text of ["$100", "cost$", "\\$code-review", "$review 已经结束"]) assert.equal(skillMention(text, text.length), null);
});
