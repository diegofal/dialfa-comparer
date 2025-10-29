-- ============================================================================
-- QUERIES DE VALIDACIÓN - ROTACIÓN DE INVENTARIO
-- Base de datos: Spisa_local
-- ============================================================================
-- 
-- ⚠️ IMPORTANTE: Versión corregida (2025-10-29)
-- 
-- Problema encontrado: Las queries con LEFT JOIN duplicaban las ventas cuando
-- el filtro de fecha (FechaEmision >= DATEADD...) se aplicaba en el ON del JOIN.
-- 
-- Resultado del error: Ventas infladas (ej: 3,842 en vez de 276, factor ~14x)
-- 
-- Solución aplicada: Patrón de 2 CTEs:
--   1. Primera CTE: Agregar ventas directamente de NotaPedido_Items (INNER JOIN)
--                   con filtro de fecha en WHERE (no en ON)
--   2. Segunda CTE: LEFT JOIN del resultado con Articulos
--   3. SELECT final: Calcular métricas sobre datos ya agregados
-- 
-- Este patrón garantiza que cada venta se cuente UNA SOLA VEZ.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY 1: Verificar productos específicos de la pantalla
-- ----------------------------------------------------------------------------
-- Esta query muestra exactamente lo que calcula el sistema

-- ⚠️ VERSIÓN CORREGIDA - Agrega ventas primero, luego une con productos
WITH VentasAgregadas AS (
    -- Primero: Agregar ventas por producto (sin duplicación)
    SELECT 
        npi.IdArticulo,
        SUM(npi.Cantidad) AS TotalVendido,
        MAX(np.FechaEmision) AS UltimaVenta
    FROM NotaPedido_Items npi
    INNER JOIN NotaPedidos np ON npi.IdNotaPedido = np.IdNotaPedido
    WHERE np.FechaEmision >= DATEADD(MONTH, -12, GETDATE())
    GROUP BY npi.IdArticulo
),
VentasProducto AS (
    -- Segundo: Unir con información de productos
    SELECT 
        a.idArticulo,
        a.codigo,
        a.descripcion,
        a.cantidad AS StockActual,
        COALESCE(v.TotalVendido, 0) AS TotalVendido,
        v.UltimaVenta
    FROM Articulos a
    LEFT JOIN VentasAgregadas v ON a.idArticulo = v.IdArticulo
    WHERE a.codigo IN ('C4521/2', 'C454', 'C45EP6', 'CRL9010', 'TR2X1/2')
)
SELECT 
    codigo AS 'Código',
    descripcion AS 'Descripción',
    StockActual AS 'Stock Actual',
    TotalVendido AS 'Total Vendido (12m)',
    
    -- Promedio mensual de ventas
    TotalVendido / 12.0 AS 'Ventas/Mes Promedio',
    
    -- Meses de stock (cuánto tiempo dura el inventario)
    CASE 
        WHEN TotalVendido / 12.0 > 0 
        THEN StockActual / (TotalVendido / 12.0)
        ELSE 999
    END AS 'Meses de Stock',
    
    -- Porcentaje de rotación anual
    CASE 
        WHEN StockActual > 0 
        THEN (TotalVendido / 12.0 * 12) / StockActual * 100
        ELSE 0
    END AS 'Rotación Anual %',
    
    -- Clasificación
    CASE 
        WHEN TotalVendido = 0 THEN 'Sin Movimiento'
        WHEN (TotalVendido / 12.0 * 12) / NULLIF(StockActual, 0) * 100 >= 400 THEN 'Alta Rotación'
        WHEN (TotalVendido / 12.0 * 12) / NULLIF(StockActual, 0) * 100 >= 200 THEN 'Media Rotación'
        WHEN (TotalVendido / 12.0 * 12) / NULLIF(StockActual, 0) * 100 > 0 THEN 'Baja Rotación'
        ELSE 'Sin Movimiento'
    END AS 'Categoría Rotación',
    
    UltimaVenta AS 'Última Venta'
FROM VentasProducto
ORDER BY codigo;


-- ----------------------------------------------------------------------------
-- QUERY 2: Ver ventas detalladas de un producto específico
-- ----------------------------------------------------------------------------
-- Cambia 'C4521/2' por el código que quieras verificar

SELECT 
    np.FechaEmision AS 'Fecha Pedido',
    npi.Cantidad AS 'Cantidad Vendida',
    npi.PrecioUnitario AS 'Precio Unitario',
    np.NumeroOrden AS 'Número Orden'
FROM NotaPedido_Items npi
INNER JOIN NotaPedidos np ON npi.IdNotaPedido = np.IdNotaPedido
INNER JOIN Articulos a ON npi.IdArticulo = a.idArticulo
WHERE a.codigo = 'C4521/2'  -- ⚠️ CAMBIA ESTE CÓDIGO
  AND np.FechaEmision >= DATEADD(MONTH, -12, GETDATE())
ORDER BY np.FechaEmision DESC;


-- ----------------------------------------------------------------------------
-- QUERY 3: Resumen de ventas mensuales de un producto
-- ----------------------------------------------------------------------------
-- Ver cómo varían las ventas mes a mes

SELECT 
    a.codigo AS 'Código',
    YEAR(np.FechaEmision) AS 'Año',
    MONTH(np.FechaEmision) AS 'Mes',
    DATENAME(MONTH, np.FechaEmision) AS 'Nombre Mes',
    SUM(npi.Cantidad) AS 'Cantidad Vendida',
    COUNT(DISTINCT np.IdNotaPedido) AS 'Número Pedidos'
FROM NotaPedido_Items npi
INNER JOIN NotaPedidos np ON npi.IdNotaPedido = np.IdNotaPedido
INNER JOIN Articulos a ON npi.IdArticulo = a.idArticulo
WHERE a.codigo = 'C4521/2'  -- ⚠️ CAMBIA ESTE CÓDIGO
  AND np.FechaEmision >= DATEADD(MONTH, -12, GETDATE())
GROUP BY a.codigo, YEAR(np.FechaEmision), MONTH(np.FechaEmision), DATENAME(MONTH, np.FechaEmision)
ORDER BY YEAR(np.FechaEmision) DESC, MONTH(np.FechaEmision) DESC;


-- ----------------------------------------------------------------------------
-- QUERY 4: Top 20 productos con ALTA rotación
-- ----------------------------------------------------------------------------

WITH VentasTemp AS (
    SELECT 
        npi.IdArticulo,
        SUM(npi.Cantidad) AS TotalVendido
    FROM NotaPedido_Items npi
    INNER JOIN NotaPedidos np ON npi.IdNotaPedido = np.IdNotaPedido
    WHERE np.FechaEmision >= DATEADD(MONTH, -12, GETDATE())
    GROUP BY npi.IdArticulo
),
VentasAlta AS (
    SELECT 
        a.idArticulo,
        a.codigo,
        a.descripcion,
        a.cantidad AS Stock,
        COALESCE(v.TotalVendido, 0) AS TotalVendido
    FROM Articulos a
    LEFT JOIN VentasTemp v ON a.idArticulo = v.IdArticulo
    WHERE a.discontinuado = 0
)
SELECT TOP 20
    codigo AS 'Código',
    descripcion AS 'Descripción',
    Stock AS 'Stock',
    TotalVendido / 12.0 AS 'Ventas/Mes',
    CASE 
        WHEN Stock > 0 
        THEN ((TotalVendido / 12.0) * 12) / Stock * 100
        ELSE 0
    END AS 'Rotación %',
    'Alta Rotación' AS 'Categoría'
FROM VentasAlta
WHERE ((TotalVendido / 12.0) * 12) / NULLIF(Stock, 0) * 100 >= 400
ORDER BY ((TotalVendido / 12.0) * 12) / NULLIF(Stock, 0) * 100 DESC;


-- ----------------------------------------------------------------------------
-- QUERY 5: Distribución completa de rotación (resumen general)
-- ----------------------------------------------------------------------------

WITH VentasTemp AS (
    SELECT 
        npi.IdArticulo,
        SUM(npi.Cantidad) AS TotalVendido
    FROM NotaPedido_Items npi
    INNER JOIN NotaPedidos np ON npi.IdNotaPedido = np.IdNotaPedido
    WHERE np.FechaEmision >= DATEADD(MONTH, -12, GETDATE())
    GROUP BY npi.IdArticulo
),
VentasAgregadas AS (
    SELECT 
        a.idArticulo,
        a.codigo,
        a.cantidad AS Stock,
        COALESCE(v.TotalVendido, 0) AS TotalVendido
    FROM Articulos a
    LEFT JOIN VentasTemp v ON a.idArticulo = v.IdArticulo
    WHERE a.discontinuado = 0
),
VelocityCalc AS (
    SELECT 
        idArticulo,
        codigo,
        Stock,
        TotalVendido / 12.0 AS VentasMes,
        CASE 
            WHEN Stock > 0 
            THEN ((TotalVendido / 12.0) * 12) / Stock * 100
            ELSE 0
        END AS RotacionPct,
        CASE 
            WHEN TotalVendido = 0 THEN 'Sin Movimiento'
            WHEN ((TotalVendido / 12.0) * 12) / NULLIF(Stock, 0) * 100 >= 400 THEN 'Alta Rotación'
            WHEN ((TotalVendido / 12.0) * 12) / NULLIF(Stock, 0) * 100 >= 200 THEN 'Media Rotación'
            WHEN ((TotalVendido / 12.0) * 12) / NULLIF(Stock, 0) * 100 > 0 THEN 'Baja Rotación'
            ELSE 'Sin Movimiento'
        END AS Categoria
    FROM VentasAgregadas
)
SELECT 
    Categoria AS 'Categoría de Rotación',
    COUNT(*) AS 'Cantidad Productos',
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,2)) AS '% del Total'
FROM VelocityCalc
GROUP BY Categoria
ORDER BY 
    CASE Categoria
        WHEN 'Alta Rotación' THEN 1
        WHEN 'Media Rotación' THEN 2
        WHEN 'Baja Rotación' THEN 3
        WHEN 'Sin Movimiento' THEN 4
    END;


-- ----------------------------------------------------------------------------
-- QUERY 6: Productos con más de X meses de inventario
-- ----------------------------------------------------------------------------
-- Encuentra productos que tienen exceso de stock (>12 meses)

WITH VentasTemp AS (
    SELECT 
        npi.IdArticulo,
        SUM(npi.Cantidad) AS TotalVendido
    FROM NotaPedido_Items npi
    INNER JOIN NotaPedidos np ON npi.IdNotaPedido = np.IdNotaPedido
    WHERE np.FechaEmision >= DATEADD(MONTH, -12, GETDATE())
    GROUP BY npi.IdArticulo
),
VentasExceso AS (
    SELECT 
        a.idArticulo,
        a.codigo,
        a.descripcion,
        a.cantidad AS StockActual,
        a.preciounitario,
        COALESCE(v.TotalVendido, 0) AS TotalVendido
    FROM Articulos a
    LEFT JOIN VentasTemp v ON a.idArticulo = v.IdArticulo
    WHERE a.discontinuado = 0 AND a.cantidad > 0
)
SELECT TOP 50
    codigo AS 'Código',
    descripcion AS 'Descripción',
    StockActual AS 'Stock Actual',
    TotalVendido / 12.0 AS 'Ventas/Mes',
    CASE 
        WHEN TotalVendido / 12.0 > 0 
        THEN StockActual / (TotalVendido / 12.0)
        ELSE 999
    END AS 'Meses de Stock',
    preciounitario AS 'Precio Unit.',
    StockActual * preciounitario AS 'Valor Stock Total'
FROM VentasExceso
WHERE TotalVendido / 12.0 > 0
  AND StockActual / (TotalVendido / 12.0) > 12  -- Más de 12 meses
ORDER BY StockActual / (TotalVendido / 12.0) DESC;


-- ----------------------------------------------------------------------------
-- QUERY 7: Validación matemática detallada de C4521/2
-- ----------------------------------------------------------------------------
-- Query super detallada para entender cada paso del cálculo

DECLARE @Codigo VARCHAR(50) = 'C4521/2';  -- ⚠️ CAMBIA ESTE CÓDIGO

WITH VentasTemp AS (
    SELECT 
        npi.IdArticulo,
        SUM(npi.Cantidad) AS TotalVendido
    FROM NotaPedido_Items npi
    INNER JOIN NotaPedidos np ON npi.IdNotaPedido = np.IdNotaPedido
    WHERE np.FechaEmision >= DATEADD(MONTH, -12, GETDATE())
    GROUP BY npi.IdArticulo
),
VentasCalculo AS (
    SELECT 
        a.codigo,
        a.cantidad AS StockActual,
        COALESCE(v.TotalVendido, 0) AS TotalVendido
    FROM Articulos a
    LEFT JOIN VentasTemp v ON a.idArticulo = v.IdArticulo
    WHERE a.codigo = @Codigo
)
SELECT 
    codigo AS 'Código Producto',
    '---' AS '---',
    
    -- Paso 1: Stock actual
    StockActual AS '1. Stock Actual',
    
    -- Paso 2: Total vendido en 12 meses
    TotalVendido AS '2. Total Vendido (12m)',
    
    -- Paso 3: Ventas promedio por mes
    TotalVendido / 12.0 AS '3. Ventas/Mes (Total ÷ 12)',
    
    -- Paso 4: Ventas anualizadas
    (TotalVendido / 12.0) * 12 AS '4. Ventas Anualizadas (Ventas/Mes × 12)',
    
    -- Paso 5: Rotación como decimal
    CASE 
        WHEN StockActual > 0 
        THEN ((TotalVendido / 12.0) * 12) / StockActual
        ELSE 0
    END AS '5. Rotación Decimal (Anual ÷ Stock)',
    
    -- Paso 6: Rotación como porcentaje
    CASE 
        WHEN StockActual > 0 
        THEN ((TotalVendido / 12.0) * 12) / StockActual * 100
        ELSE 0
    END AS '6. Rotación % (Decimal × 100)',
    
    -- Paso 7: Meses de stock
    CASE 
        WHEN TotalVendido / 12.0 > 0 
        THEN StockActual / (TotalVendido / 12.0)
        ELSE 999
    END AS '7. Meses de Stock (Stock ÷ Ventas/Mes)',
    
    -- Paso 8: Clasificación
    CASE 
        WHEN TotalVendido = 0 THEN 'Sin Movimiento (0%)'
        WHEN ((TotalVendido / 12.0) * 12) / NULLIF(StockActual, 0) * 100 >= 400 
            THEN 'Alta Rotación (≥400%)'
        WHEN ((TotalVendido / 12.0) * 12) / NULLIF(StockActual, 0) * 100 >= 200 
            THEN 'Media Rotación (200-399%)'
        WHEN ((TotalVendido / 12.0) * 12) / NULLIF(StockActual, 0) * 100 > 0 
            THEN 'Baja Rotación (<200%)'
        ELSE 'Sin Movimiento'
    END AS '8. Categoría Final'
FROM VentasCalculo;


-- ----------------------------------------------------------------------------
-- QUERY 8: Ver todos los pedidos que tienen un producto específico
-- ----------------------------------------------------------------------------

SELECT 
    np.NumeroOrden AS 'Nro. Orden',
    np.FechaEmision AS 'Fecha',
    FORMAT(np.FechaEmision, 'yyyy-MM') AS 'Año-Mes',
    npi.Cantidad AS 'Cantidad',
    npi.PrecioUnitario AS 'Precio Unit.',
    npi.Cantidad * npi.PrecioUnitario AS 'Total Línea',
    DATEDIFF(DAY, np.FechaEmision, GETDATE()) AS 'Días Atrás'
FROM NotaPedidos np
INNER JOIN NotaPedido_Items npi ON np.IdNotaPedido = npi.IdNotaPedido
INNER JOIN Articulos a ON npi.IdArticulo = a.idArticulo
WHERE a.codigo = 'C4521/2'  -- ⚠️ CAMBIA ESTE CÓDIGO
  AND np.FechaEmision >= DATEADD(MONTH, -12, GETDATE())
ORDER BY np.FechaEmision DESC;


-- ============================================================================
-- INSTRUCCIONES DE USO:
-- ============================================================================
-- 
-- 1. Copia y pega cada query en SQL Server Management Studio o Azure Data Studio
-- 2. Para queries con parámetros, busca el comentario "⚠️ CAMBIA ESTE CÓDIGO"
-- 3. Los resultados deben coincidir EXACTAMENTE con lo que muestra la web
-- 
-- QUERY RECOMENDADA PARA EMPEZAR: #1 (productos específicos de pantalla)
-- 
-- ============================================================================

