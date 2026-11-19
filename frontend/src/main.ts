import "./styles.css";
import { makeApi } from "./api";
import { mountApp } from "./app";

mountApp(document.getElementById("root")!, makeApi());
