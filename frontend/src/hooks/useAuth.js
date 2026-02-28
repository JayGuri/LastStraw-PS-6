import { useState, useCallback, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
const TOKEN_KEY = "hackx_auth_token";
const USER_KEY = "hackx_user";

/**
 * Hook for authentication state and operations
 * Manages JWT tokens and user info in localStorage
 */
export function useAuth() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Initialize from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem(TOKEN_KEY);
    const storedUser = localStorage.getItem(USER_KEY);

    if (storedToken) {
      setToken(storedToken);
      if (storedUser) {
        try {
          setUser(JSON.parse(storedUser));
        } catch (e) {
          console.error("Failed to parse stored user:", e);
          localStorage.removeItem(USER_KEY);
        }
      }
    }
  }, []);

  /**
   * Store authentication data in localStorage
   */
  const storeAuthData = useCallback((newToken, newUser) => {
    localStorage.setItem(TOKEN_KEY, newToken);
    localStorage.setItem(USER_KEY, JSON.stringify(newUser));
    setToken(newToken);
    setUser(newUser);
  }, []);

  /**
   * Clear authentication data
   */
  const clearAuthData = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setToken(null);
    setUser(null);
  }, []);

  /**
   * Get the current authorization header
   */
  const getAuthHeader = useCallback(() => {
    if (token) {
      return { Authorization: `Bearer ${token}` };
    }
    return {};
  }, [token]);

  /**
   * Login with email and password
   */
  const login = useCallback(
    async (email, password) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(`${API_URL}/auth/login`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ email, password }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(
            errorData.detail || errorData.error || "Login failed",
          );
        }

        const data = await response.json();
        storeAuthData(data.token, data.user);
        return data;
      } catch (err) {
        const message = err.message || "An error occurred during login";
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [storeAuthData],
  );

  /**
   * Logout and invalidate token
   */
  const logout = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      if (token) {
        await fetch(`${API_URL}/auth/logout`, {
          method: "POST",
          headers: {
            ...getAuthHeader(),
            "Content-Type": "application/json",
          },
        });
      }
    } catch (err) {
      console.error("Logout error:", err);
    } finally {
      clearAuthData();
      setIsLoading(false);
    }
  }, [token, getAuthHeader, clearAuthData]);

  /**
   * Get current user info
   */
  const getCurrentUser = useCallback(async () => {
    if (!token) {
      return null;
    }

    try {
      const response = await fetch(`${API_URL}/auth/me`, {
        method: "GET",
        headers: getAuthHeader(),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch user info");
      }

      const data = await response.json();
      setUser(data.user);
      localStorage.setItem(USER_KEY, JSON.stringify(data.user));
      return data.user;
    } catch (err) {
      console.error("Get user error:", err);
      // Token might be invalid
      if (response?.status === 401) {
        clearAuthData();
      }
      return null;
    }
  }, [token, getAuthHeader, clearAuthData]);

  /**
   * Refresh authentication token
   */
  const refreshToken = useCallback(async () => {
    if (!token) {
      return null;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/auth/refresh-token`, {
        method: "POST",
        headers: getAuthHeader(),
      });

      if (!response.ok) {
        throw new Error("Token refresh failed");
      }

      const data = await response.json();
      storeAuthData(data.token, data.user);
      return data;
    } catch (err) {
      console.error("Token refresh error:", err);
      clearAuthData();
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [token, getAuthHeader, storeAuthData, clearAuthData]);

  /**
   * Initiate Google OAuth login
   */
  const loginWithGoogle = useCallback(() => {
    // Redirect to backend's Google OAuth endpoint
    window.location.href = `${API_URL}/auth/google`;
  }, []);

  /**
   * Handle OAuth callback with token in URL
   */
  const handleOAuthCallback = useCallback(() => {
    const params = new URLSearchParams(window.location.search);
    const tokenFromUrl = params.get("token");

    if (tokenFromUrl) {
      // Decode token to get user info (basic decoding without verification)
      try {
        const parts = tokenFromUrl.split(".");
        if (parts.length === 3) {
          const payload = JSON.parse(atob(parts[1]));
          const userInfo = {
            id: payload.user_id,
            email: payload.email,
            subscription_level: payload.subscription_level,
            auth_provider: payload.auth_provider,
          };
          storeAuthData(tokenFromUrl, userInfo);

          // Clean up URL
          window.history.replaceState(
            {},
            document.title,
            window.location.pathname,
          );

          return { token: tokenFromUrl, user: userInfo };
        }
      } catch (e) {
        console.error("Failed to decode OAuth token:", e);
      }
    }

    return null;
  }, [storeAuthData]);

  return {
    // State
    user,
    token,
    isLoading,
    error,
    isAuthenticated: !!token,

    // Methods
    login,
    logout,
    getCurrentUser,
    refreshToken,
    loginWithGoogle,
    handleOAuthCallback,
    getAuthHeader,
    storeAuthData,
    clearAuthData,
  };
}
