pm.test("200 OK", () => {
  pm.response.to.have.status(200);
});

pm.test("items is array", () => {
  const json = pm.response.json();
  pm.expect(json).to.have.property("items");
  pm.expect(json.items).to.be.an("array");
});
