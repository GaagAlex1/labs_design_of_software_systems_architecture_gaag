pm.test("422 Unprocessable Entity", () => {
  pm.response.to.have.status(422);
});

pm.test("Error payload exists (or at least non-empty body)", () => {
  const text = pm.response.text();
  pm.expect(text).to.not.equal("");
});