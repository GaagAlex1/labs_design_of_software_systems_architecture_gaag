pm.test("204 No Content", () => {
  pm.response.to.have.status(204);
});

pm.test("Empty body", () => {
  const t = pm.response.text();
  pm.expect(t === "" || t === null).to.be.true;
});
