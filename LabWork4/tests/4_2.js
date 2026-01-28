pm.test("404 Not Found (unknown job)", () => {
  pm.response.to.have.status(404);
});

pm.test("Body contains error info (non-empty)", () => {
  pm.expect(pm.response.text()).to.not.equal("");
});
