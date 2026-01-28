pm.test("200 OK", () => {
  pm.response.to.have.status(200);
});

pm.test("repo_uuid equals repo_id", () => {
  const json = pm.response.json();
  pm.expect(json).to.have.property("repo_uuid", pm.environment.get("repo_id"));
});
