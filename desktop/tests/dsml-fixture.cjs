function encodeDSML(calls) {
  return "开始执行。\n\n<｜DSML｜tool_calls>\n" + calls.map((call) => '<｜DSML｜invoke name="' + call.name + '">\n' +
    Object.entries(call.input).map(([name, value]) => '<｜DSML｜parameter name="' + name + '" string="' + (typeof value === "string") + '">' +
      (typeof value === "string" ? value : JSON.stringify(value)) + '</｜DSML｜parameter>').join("\n") + '\n</｜DSML｜invoke>').join("\n") + "\n</｜DSML｜tool_calls>";
}
module.exports = { encodeDSML };
