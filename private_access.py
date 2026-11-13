"""Optional password gate for private ZicZac deployments.

Prefer deployment-level access controls as the primary protection. The optional
secret is never stored in the repository, query string or prediction database.
"""
import hmac
import os


def matches(candidate, expected):
    """Use constant-time equality for secret comparisons."""
    return bool(expected) and hmac.compare_digest(
        str(candidate).encode("utf-8"), str(expected).encode("utf-8")
    )


def require_access(st):
    """Protect all app screens when the owner's Streamlit Secret is configured.

    Returns True when password protection is configured. If absent, do not
    silently claim the site is private or disrupt an existing deployment.
    """
    try:
        secret = st.secrets.get("ZICZAC_ACCESS_PASSWORD", "")
    except (AttributeError, OSError, FileNotFoundError):
        secret = ""
    secret = str(secret or os.environ.get("ZICZAC_ACCESS_PASSWORD", ""))
    if not secret:
        return False
    if st.session_state.get("ziczac_access_granted") is True:
        return True

    st.markdown("## 🔐 ZicZac Predict AI · Private Login")
    st.caption("Only enter the password configured by the owner in Streamlit Secrets.")
    failures = int(st.session_state.get("ziczac_access_failures", 0))
    if failures >= 5:
        st.error("Too many unsuccessful attempts in this session. Close the tab and contact the owner.")
        st.stop()
    with st.form("ziczac_private_login", clear_on_submit=True):
        password = st.text_input("Access password", type="password")
        submitted = st.form_submit_button("Unlock ZicZac", type="primary", use_container_width=True)
    if submitted:
        if matches(password, secret):
            st.session_state["ziczac_access_granted"] = True
            st.session_state["ziczac_access_failures"] = 0
            st.rerun()
        else:
            st.session_state["ziczac_access_failures"] = failures + 1
            st.error("Incorrect password.")
    st.stop()
