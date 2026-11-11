
import { useState } from "react";
import { loginAdmin, getAdminSession, setAdminToken } from "../../api/rescueway";

export default function AdminLogin({ onAuthenticated }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const result = await loginAdmin(username.trim(), password);
      setAdminToken(result.access_token);
      const session = await getAdminSession();

      if (!session.authenticated) {
        throw new Error("Admin session verification failed.");
      }

      onAuthenticated(session.username);
    } catch (err) {
      setAdminToken(null);
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Login failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-5 py-10"
      style={{ background: "#0f1117" }}>
      <div className="w-full max-w-md rounded-2xl border border-[#2a2d3a]
        bg-[#1a1d27] p-8 shadow-2xl">

        <div className="text-center mb-8">
          <div className="mx-auto mb-5 flex h-16 w-16 items-center
            justify-center rounded-2xl bg-blue-500/10 text-3xl">
            🚦
          </div>

          <h1 className="text-2xl font-bold text-white">
            RescueWay
          </h1>

          <p className="mt-2 text-sm text-[#9ca0b0]">
            Administrator Portal
          </p>
        </div>

        <div className="mb-6 rounded-lg border border-blue-500/20
          bg-blue-500/5 p-3">
          <p className="text-xs text-blue-300 text-center">
            Restricted access · Authorized administrators only
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="mb-2 block text-sm text-[#c5c8d3]">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              className="w-full rounded-lg border border-[#343849]
                bg-[#0f1117] px-4 py-3 text-white outline-none
                focus:border-blue-500"
              placeholder="Enter administrator username"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm text-[#c5c8d3]">
              Password
            </label>

            <div className="flex rounded-lg border border-[#343849]
              bg-[#0f1117] focus-within:border-blue-500">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                required
                className="min-w-0 flex-1 bg-transparent px-4 py-3
                  text-white outline-none"
                placeholder="Enter your password"
              />

              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="px-4 text-xs text-blue-400"
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </div>

          {error && (
            <div role="alert" className="rounded-lg border border-red-500/30
              bg-red-500/10 p-3 text-sm text-red-300">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-blue-600 py-3
              font-semibold text-white transition hover:bg-blue-500
              disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Authenticating..." : "Sign in to Admin Dashboard"}
          </button>
        </form>

        <p className="mt-7 text-center text-xs text-[#6b7080]">
          RescueWay · Emergency Traffic Prioritization
        </p>
      </div>
    </div>
  );
}
