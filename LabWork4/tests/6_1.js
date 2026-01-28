pm.test("200 OK", () => {
  pm.response.to.have.status(200);
});

pm.test("answer is non-empty string", () => {
  const json = pm.response.json();
  pm.expect(json).to.have.property("answer");
  pm.expect(json.answer).to.be.a("string").and.not.empty;
});
