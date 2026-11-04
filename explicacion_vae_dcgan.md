# Guía Completa: VAE vs DCGAN y Análisis de Código

---

## 1. Conceptos Fundamentales: VAE vs DCGAN

Tanto los **VAE** (*Variational Autoencoders*) como las **DCGAN** (*Deep Convolutional Generative Adversarial Networks*) pertenecen a la familia de los **modelos generativos profundos**, cuyo objetivo es aprender la distribución de datos subyacente para sintetizar nuevas muestras (en este caso, imágenes de caracteres/K-MNIST).

```mermaid
flowchart TD
    subgraph VAE ["VAE (Autoencoder Variacional)"]
        direction LR
        X[Imagen X] --> Enc[Encoder]
        Enc --> Dist["μ y log(σ²)"]
        Dist --> Sample["Muestreo z ~ N(μ, σ²)"]
        Sample --> Dec[Decoder]
        Dec --> X_hat["Imagen Reconstruida X̂"]
    end

    subgraph DCGAN ["DCGAN (Red Generativa Adversaria)"]
        direction LR
        Z[Ruido z ~ N(0, I)] --> G[Generador]
        G --> Fake[Imagen Sintética]
        Real[Imagen Real] --> D[Discriminador]
        Fake --> D
        D --> Pred["Real (1) vs Falsa (0)"]
    end
```

---

### ¿Qué es un VAE (Variational Autoencoder)?
Un VAE combina **redes neuronales** con **inferencia variacional probabilística**.

- **Encoder (Red de Inferencia):** Modela la distribución aproximada $q_\phi(z|x)$. En vez de comprimir una imagen en un vector latente determinista, predice los parámetros de una distribución gaussiana: la media $\mu$ y el logaritmo de la varianza $\log(\sigma^2)$.
- **Reparameterization Trick:** Para permitir la retropropagación del gradiente a través del muestreo estocástico, se muestrea ruido externo $\epsilon \sim \mathcal{N}(0, I)$ y se calcula:
  $$z = \mu + \sigma \odot \epsilon$$
- **Decoder (Red Generativa):** Modela $p_\theta(x|z)$, recibiendo el vector latente $z$ para reconstruir la imagen original.
- **Función Objetivo (ELBO):** Maximiza el *Evidence Lower Bound*:
  $$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \parallel p(z))$$
  - Primer término: Fidelidad de reconstrucción (Binary Cross-Entropy o MSE).
  - Segundo término: Regularizador (Divergencia de Kullback-Leibler) que obliga al espacio latente a parecerse a una normal estándar $\mathcal{N}(0, I)$.

---

### ¿Qué es un DCGAN?
Un DCGAN aplica **convoluciones profundas** a la arquitectura de una GAN (Red Generativa Adversaria), enfrentando dos redes en un juego de suma cero minimax:

- **Generador ($G$):** Toma un vector latente aleatorio $z \sim \mathcal{N}(0, I)$ y, mediante convoluciones transpuestas, genera una imagen intentando engañar al discriminador.
- **Discriminador ($D$):** Una red convolucional estándar que clasifica si una imagen de entrada proviene del dataset real o fue inventada por el generador.
- **Objetivo Minimax:**
  $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

---

### Tabla Comparativa: VAE vs DCGAN

| Característica | VAE (Variational Autoencoder) | DCGAN (Deep Convolutional GAN) |
| :--- | :--- | :--- |
| **Principio Teórico** | Inferencia Bayesiana / Máxima Verosimilitud (ELBO) | Teoría de juegos Minimax (Adversarial) |
| **Arquitectura** | **Encoder + Decoder** | **Generador + Discriminador** |
| **Mapeo Inverso ($X \to Z$)** | **Sí nativo**: el Encoder proyecta imágenes reales a coordenadas latentes. | **No nativo**: requiere técnicas adicionales (como BiGAN o inversión latente por optimización). |
| **Calidad Visual** | **Algo borrosa**: las pérdidas a nivel de píxel (BCE/MSE) promedian modos posibles. | **Muy nítida**: el discriminador penaliza detalles poco realistas a nivel local y global. |
| **Estructura del Espacio Latente** | **Continua, suave y densa**: ideal para interpolaciones, clustering y manipulación de atributos. | Puede presentar desconexiones, vacíos o colapso de modos (*mode collapse*). |
| **Estabilidad de Entrenamiento** | **Muy alta**: optimiza una única función de coste convergente. | **Baja / Delicada**: requiere equilibrar las tasas de aprendizaje y arquitecturas de $G$ y $D$. |
| **Función de Pérdida** | Reconstrucción (BCE) + Regularización KL | Cross-Entropy Binaria adversaria (o Wasserstein) |

---

## 2. Explicación Detallada del Código

El script implementa un **VAE Convolucional** con espacio latente bidimensional (`LATENT_DIM = 2`) para imágenes monocromáticas $28 \times 28$ (típicamente K-MNIST / MNIST).

### Bloque 1: Arquitectura `ConvVAE`

```python
LATENT_DIM = 2

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # --- ENCODER: Reduce resolución espacial y extrae características ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),    # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # Mantiene 7x7
            nn.Flatten(),                                                       # Vector de 128*7*7 = 6272
        )
        # --- CABEZAS LATENTES: Parámetros gaussianos ---
        self.mu = nn.Linear(128 * 7 * 7, latent_dim)        # Vector de medias μ
        self.logvar = nn.Linear(128 * 7 * 7, latent_dim)    # Vector de log-varianzas log(σ²)

        # --- DECODER: Reconstrucción espacial ---
        self.decoder_input = nn.Linear(latent_dim, 128 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),                                       # Vector -> Tensor (128, 7, 7)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 7x7 -> 14x14
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 14x14 -> 28x28
            nn.Conv2d(16, 1, kernel_size=3, padding=1), nn.Sigmoid(),                   # Escala intensidades [0, 1]
        )
```

#### ¿Por qué predecir `logvar` en lugar de la varianza $\sigma^2$?
La varianza debe ser estrictamente positiva ($\sigma^2 > 0$). Si la red predijera directamente $\sigma^2$, una salida negativa rompería el cálculo. Al predecir $\log(\sigma^2)$, el rango de la capa lineal puede ser cualquier número real $\in (-\infty, \infty)$. La desviación estándar se recupera fácilmente con:
$$\sigma = \exp(0.5 \cdot \log(\sigma^2))$$

#### Métodos de la Red y el Truco de Reparametrización
```python
    def encode(self, images):
        encoded = self.encoder(images)
        return self.mu(encoded), self.logvar(encoded)

    def decode(self, latent):
        return self.decoder(self.decoder_input(latent))

    def forward(self, images, sample=True):
        mu, logvar = self.encode(images)
        if sample:
            # Reparameterization Trick: z = μ + ε * σ
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            # Evaluación determinista usando la media central
            latent = mu
        return self.decode(latent), mu, logvar
```

---

### Bloque 2: Función de Pérdida (`vae_components`)

```python
def vae_components(model, images, sample):
    reconstruction, mu, logvar = model(images, sample=sample)
    
    # 1. Pérdida de Reconstrucción (BCE entre píxeles reales y reconstruidos)
    reconstruction_loss = F.binary_cross_entropy(reconstruction, images, reduction="sum")
    
    # 2. Divergencia KL analítica con la Normal estándar N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Pérdida total promediada por imagen
    total = (reconstruction_loss + kl_loss) / len(images)
    return total, reconstruction_loss / len(images), kl_loss / len(images), reconstruction
```

- **`reconstruction_loss`**: Mide qué tan parecida es la imagen reconstruida a la original píxel por píxel.
- **`kl_loss`**: Fórmula cerrada para la divergencia entre dos normales univariadas $\mathcal{N}(\mu, \sigma^2)$ y $\mathcal{N}(0, 1)$:
  $$D_{\text{KL}} = -\frac{1}{2} \sum_{i=1}^d \left( 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2 \right)$$
  Evita que la red colapse a un autoencoder determinista estándar y agrupa los datos cerca del origen de forma suave.

---

### Bloque 3: Bucle de Entrenamiento y Validación

1. **Fase de Entrenamiento (`train`)**:
   - Activa `sample=True` para inyectar ruido gaussiano y forzar al modelo a ser robusto en todo el vecindario de $\mu$.
   - Realiza `optimizer.zero_grad(set_to_none=True)` (más eficiente en memoria que poner ceros).
   - Calcula gradientes con `total.backward()` y actualiza pesos con Adam.
2. **Fase de Validación / Dev (`train(False)` / `torch.no_grad()`)**:
   - Usa `sample=False` para obtener la mejor reconstrucción determinista ($z = \mu$).
   - Computa tanto las pérdidas teóricas (ELBO, BCE, KL) como el **MSE** (*Mean Squared Error*) normalizado por imagen y píxeles ($28 \times 28$).

---

### Bloque 4: Análisis Visual y Métricas

El código genera cuatro visualizaciones clave:

1. **Curvas de Aprendizaje (3 subplots):**
   - **ELBO:** Pérdida total combinada.
   - **Reconstrucción BCE:** Comprueba cómo mejora la nitidez y fidelidad con las épocas.
   - **Divergencia KL:** Muestra cuánta información está almacenando el espacio latente sin alejarse del prior.
2. **Originales vs Reconstruidas (10 pares):**
   - Muestra las imágenes reales en la fila superior y la reconstrucción producida por el decoder en la fila inferior.
3. **Plano Latente 2D ($z_1, z_2$):**
   - Al tener `LATENT_DIM = 2`, cada imagen de validación se mapea a un punto $(z_1, z_2)$.
   - Se plotean con diferentes colores según la clase (los 10 caracteres kanji de K-MNIST). Permite ver si el VAE agrupó los símbolos parecidos de forma no supervisada.
4. **Muestreo del Prior e Interpolación:**
   - **Muestras del Prior:** Genera 10 imágenes nuevas alimentando al decoder con ruido aleatorio estándar $z \sim \mathcal{N}(0, I)$, demostrando la capacidad puramente generativa.
   - **Interpolación Lineal:** Toma las coordenadas latentes de dos caracteres diferentes ($z_A$ y $z_B$) y decodifica puntos intermedios:
     $$z(\alpha) = (1 - \alpha) z_A + \alpha z_B \quad \text{con } \alpha \in [0, 1]$$
     Permite observar cómo un carácter kanji se transforma de manera continua y fluida en otro sin saltos abruptos.

