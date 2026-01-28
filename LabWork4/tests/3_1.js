pm.test("202 Accepted", () => {
  pm.response.to.have.status(202);
});

pm.test("Save job_id from job_uuid", () => {
  const json = pm.response.json();
  pm.expect(json).to.have.property("job_uuid");
  pm.environment.set("job_id", json.job_uuid);
});
